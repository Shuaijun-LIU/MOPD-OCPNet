import requests
import json

def fetch_noaa_data(station_id):
    """
    Fetch data from NOAA Tides and Currents API for a specific station ID.
    :param station_id: NOAA station ID
    """
    API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    # Base parameters
    base_params = {
        'date': 'latest',
        'station': station_id,
        'units': 'metric',
        'time_zone': 'gmt',
        'format': 'json',
        'application': 'my_app'
    }

    # Products to fetch
    products = ['water_level', 'currents', 'wind', 'air_temperature', 'water_temperature', 'air_pressure']
    field_mappings = {'t': 'Timestamp', 'v': 'Value', 's': 'Speed', 'd': 'Direction', 'q': 'Quality'}
    summary = {'success': [], 'error': []}  # Track success and errors

    for product in products:
        params = base_params.copy()
        params['product'] = product
        if product == 'currents':
            params.pop('datum', None)  # 'datum' not needed for currents
        else:
            params['datum'] = 'MLLW'

        print(f"\nFetching data for product: {product}...")
        try:
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if 'data' in data:
                summary['success'].append(product)
                print(f"Results for {product}:")
                for record in data['data']:
                    output = {field_mappings.get(k, k): v for k, v in record.items()}
                    if product == 'water_level':
                        print(f"{output['Timestamp']}: Water Level = {output['Value']} meters")
                    elif product == 'currents':
                        print(f"{output['Timestamp']}: Current Speed = {output['Speed']} knots, "
                              f"Direction = {output['Direction']}°")
                    elif product == 'wind':
                        print(f"{output['Timestamp']}: Wind Speed = {output['Speed']} m/s, "
                              f"Direction = {output['Direction']}°")
                    elif product == 'air_temperature':
                        print(f"{output['Timestamp']}: Air Temperature = {output['Value']}°C")
                    elif product == 'water_temperature':
                        print(f"{output['Timestamp']}: Water Temperature = {output['Value']}°C")
                    elif product == 'air_pressure':
                        print(f"{output['Timestamp']}: Air Pressure = {output['Value']} hPa")
            elif 'error' in data:
                summary['error'].append(product)
                print(f"No data found for {product}: {data['error']['message']}")

        except requests.exceptions.HTTPError as http_err:
            summary['error'].append(product)
            print(f"Error fetching data for {product}: {http_err}")
        except Exception as err:
            summary['error'].append(product)
            print(f"Unexpected error for {product}: {err}")

    # Summary
    print("=====================================================")
    print("\nSummary:")
    print(f"Successful products: {', '.join(summary['success'])}")
    print(f"Failed products: {', '.join(summary['error'])}")
    print("=====================================================")
