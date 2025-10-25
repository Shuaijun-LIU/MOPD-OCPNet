import requests
import json
from math import radians, cos, sin, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    """
    R = 6371.0  # Earth radius in kilometers

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def find_nearest_current_station(user_lat, user_lon):
    """
    Find the nearest NOAA station with current data to the specified latitude and longitude.
    """
    # NOAA CO-OPS API endpoint for active current stations
    api_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=currents&status=active"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        stations = response.json().get('stations', [])

        if not stations:
            print("No active current stations found.")
            return None

        nearest_station = None
        min_distance = float('inf')

        for station in stations:
            station_id = station.get('id')
            station_name = station.get('name')
            station_lat = float(station.get('lat'))
            station_lon = float(station.get('lng'))

            distance = haversine_distance(user_lat, user_lon, station_lat, station_lon)

            if distance < min_distance:
                min_distance = distance
                nearest_station = {
                    'id': station_id,
                    'name': station_name,
                    'latitude': station_lat,
                    'longitude': station_lon,
                    'distance_km': distance
                }

        if nearest_station:
            print(f"Nearest Station ID: {nearest_station['id']}")
            print(f"Station Name: {nearest_station['name']}")
            print(f"Location: Latitude = {nearest_station['latitude']}, Longitude = {nearest_station['longitude']}")
            print(f"Distance: {nearest_station['distance_km']:.2f} km")
            print("=====================================================")
            return nearest_station['id']
        else:
            print("No nearby current stations found.")
            print("=====================================================")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching station data: {e}")
        return None

def get_nearest_station_id(latitude, longitude):
    """
    Fetch the nearest NOAA station ID based on given latitude and longitude.
    :param latitude: Latitude of the location
    :param longitude: Longitude of the location
    :return: Nearest station ID and other metadata
    """
    API_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"

    try:
        # Send request to get all stations
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()

        # Find the nearest station
        nearest_station = None
        min_distance = float('inf')

        for station in data['stations']:
            station_lat = float(station['lat'])
            station_lon = float(station['lng'])

            # Calculate distance using the Haversine formula (or a simple approximation)
            distance = ((latitude - station_lat) ** 2 + (longitude - station_lon) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                nearest_station = station

        if nearest_station:
            print(f"Location: Latitude = {nearest_station['lat']}, Longitude = {nearest_station['lng']}")
            print(f"Nearest Station ID: {nearest_station['id']}")
            print(f"Station Name: {nearest_station['name']}")
            print("=====================================================")
            return nearest_station['id']
        else:
            print("No stations found.")
            print("=====================================================")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching station list: {e}")
        return None


