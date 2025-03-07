/**
 * Main JavaScript file for Combined Ocean Environment Dataset website
 * Author: Dataset Team
 * Version: 1.0
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initializeNavigation();
    initializeDownloadTracking();
    initializeSmoothScroll();
});

/**
 * Handles navigation highlighting based on current page
 */
function initializeNavigation() {
    // Get current page path
    const currentPath = window.location.pathname;
    const currentPage = currentPath.split('/').pop() || 'index.html';
    
    // Find all navigation links
    const navLinks = document.querySelectorAll('nav ul li a');
    
    // Remove active class from all links
    navLinks.forEach(link => {
        link.classList.remove('active');
        
        // Add active class to current page link
        if (link.getAttribute('href') === currentPage) {
            link.classList.add('active');
        }
    });
}

/**
 * Tracks download clicks for analytics
 */
function initializeDownloadTracking() {
    // Find all download buttons
    const downloadButtons = document.querySelectorAll('.download-btn');
    
    // Add click event listeners
    downloadButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Get download type from parent element
            const downloadType = this.closest('.download-card').querySelector('h3').textContent;
            
            // Log download event (can be replaced with actual analytics)
            console.log(`Download clicked: ${downloadType}`);
            
            // Here you would typically send an analytics event
            // Example: gtag('event', 'download', { 'type': downloadType });
        });
    });
}

/**
 * Enables smooth scrolling for anchor links
 */
function initializeSmoothScroll() {
    // Find all links that point to an anchor
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    // Add click event listeners
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Prevent default anchor click behavior
            e.preventDefault();
            
            // Get the target element
            const targetId = this.getAttribute('href');
            if (targetId === '#') return; // Skip if href is just "#"
            
            const targetElement = document.querySelector(targetId);
            
            // Scroll to target element smoothly
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Displays a notification message
 * @param {string} message - The message to display
 * @param {string} type - The type of notification (success, error, info)
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add close button
    const closeButton = document.createElement('button');
    closeButton.innerHTML = '&times;';
    closeButton.className = 'notification-close';
    closeButton.addEventListener('click', function() {
        document.body.removeChild(notification);
    });
    
    notification.appendChild(closeButton);
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (document.body.contains(notification)) {
            document.body.removeChild(notification);
        }
    }, 5000);
}

/**
 * Validates email format
 * @param {string} email - The email to validate
 * @return {boolean} - Whether the email is valid
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Export functions for potential use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showNotification,
        isValidEmail
    };
}
