// Chart colors for different genres
const chartColors = {
    'rock': '#FF9F40',     // Orange
    'pop': '#FF6384',      // Pink
    'hip hop': '#C9CBCF',  // Gray
    'country': '#36A2EB',  // Blue
    'jazz': '#4BC0C0',     // Teal
    'blues': '#FFCE56',    // Yellow
    'reggae': '#9966FF',   // Purple
    'metal': '#EA4335',    // Red
    'electronic': '#00C851', // Green
    'K-pop': '#6D4C41', // Brown
    'r&b': '#AA66CC',      // Purple
    'folk': '#3F729B'      // Blue-gray
};

// Genre icons (emoji representations)
const genreIcons = {
    'rock': '🎸',
    'pop': '🎤',
    'hip hop': '🎧',
    'country': '🤠',
    'jazz': '🎷',
    'blues': '🎺',
    'reggae': '🌴',
    'metal': '🤘',
    'electronic': '💻',
    'K-pop': '🎻',
    'r&b': '🎵',
    'folk': '🪕'
};

// Global variables
let genreChart = null;
let currentChartType = 'pie';
let lastResults = null;

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const classifyForm = document.getElementById('classify-form');
    const resultSection = document.getElementById('results-section');
    const loadingCard = document.getElementById('loading-card');
    const errorCard = document.getElementById('error-card');
    const predictedGenre = document.getElementById('predicted-genre');
    const rfPrediction = document.getElementById('rf-prediction');
    const gbPrediction = document.getElementById('gb-prediction');
    const lrPrediction = document.getElementById('lr-prediction');
    const classifyNewBtn = document.getElementById('classify-new');
    const tryAgainBtn = document.getElementById('try-again');
    const chartRadios = document.querySelectorAll('input[name="chart-type"]');
    const genreIconElement = document.getElementById('genre-icon');
    const sampleButtons = document.querySelectorAll('.sample-lyrics');
    
    // Initialize Bootstrap modal
    let howItWorksModal = null;
    if (document.getElementById('howItWorksModal')) {
        howItWorksModal = new bootstrap.Modal(document.getElementById('howItWorksModal'));
    }
    const howItWorksCard = document.getElementById('how-it-works-card');
    const learnMoreLink = document.getElementById('learn-more-link');
    
    // Form submission
    classifyForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const lyrics = document.getElementById('lyrics').value.trim();
        const year = document.getElementById('year').value;
        
        if (lyrics === '') {
            alert('Please enter some lyrics to classify');
            return;
        }
        
        classifyLyrics(lyrics, year);
    });

    // Classify new lyrics button
    if (classifyNewBtn) {
        classifyNewBtn.addEventListener('click', function() {
            resultSection.style.display = 'none';
            document.getElementById('lyrics').value = '';
            document.getElementById('lyrics').focus();
            window.scrollTo(0, 0);
        });
    }
    
    // Try again button for error card
    if (tryAgainBtn) {
        tryAgainBtn.addEventListener('click', function() {
            errorCard.style.display = 'none';
        });
    }

    // Chart type selection
    chartRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked && lastResults) {
                currentChartType = this.value;
                updateChart(lastResults.ensemble_probabilities, lastResults.ensemble_prediction);
            }
        });
    });

    // Sample lyrics buttons
    sampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const sampleLyrics = this.getAttribute('data-lyrics');
            document.getElementById('lyrics').value = sampleLyrics;
            // Auto-submit the form
            classifyLyrics(sampleLyrics, document.getElementById('year').value);
        });
    });
    
    // Show modal when clicking on the card or learn more link
    if (howItWorksCard) {
        howItWorksCard.addEventListener('click', function() {
            if (howItWorksModal) howItWorksModal.show();
        });
    }
    
    if (learnMoreLink) {
        learnMoreLink.addEventListener('click', function(e) {
            e.preventDefault();
            if (howItWorksModal) howItWorksModal.show();
        });
    }

    // Function to classify lyrics
    function classifyLyrics(lyrics, year) {
        // Hide results and error, show loading
        resultSection.style.display = 'none';
        if (errorCard) errorCard.style.display = 'none';
        if (loadingCard) loadingCard.style.display = 'block';
        
        // Create form data
        const formData = new FormData();
        formData.append('lyrics', lyrics);
        formData.append('year', year);
        
        // For demo purposes - simulate API response with random prediction
        // Remove this mock function in production and use the actual API
        /*
        simulateClassification(lyrics)
            .then(data => {
                // Store results for chart type switching
                lastResults = data;
                
                // Update UI with results
                predictedGenre.textContent = data.ensemble_prediction;
                rfPrediction.textContent = data.model_predictions.random_forest;
                gbPrediction.textContent = data.model_predictions.gradient_boosting;
                lrPrediction.textContent = data.model_predictions.logistic_regression;
                
                // Display genre icon if available
                if (genreIconElement) {
                    const genre = data.ensemble_prediction.toLowerCase();
                    genreIconElement.innerHTML = `<span style="font-size: 3rem;">${genreIcons[genre] || '🎵'}</span>`;
                }
                
                // Create or update chart
                updateChart(data.ensemble_probabilities, data.ensemble_prediction);
                
                // Hide loading, show results
                if (loadingCard) loadingCard.style.display = 'none';
                resultSection.style.display = 'block';
                
                // Scroll to results
                resultSection.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Hide loading
                if (loadingCard) loadingCard.style.display = 'none';
                
                // Show error message
                if (errorCard) {
                    document.getElementById('error-message').textContent = error.message || 'An error occurred while classifying the lyrics.';
                    errorCard.style.display = 'block';
                } else {
                    // Otherwise use alert
                    alert('Error classifying lyrics: ' + error.message);
                }
            });
            */
        
        // /* Uncomment this when using the actual backend API /*
        fetch('/classify', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log("Received data:", data); // Debug: log the data
            
            // Store results for chart type switching
            lastResults = data;
            
            // Update UI with results
            predictedGenre.textContent = data.ensemble_prediction;
            rfPrediction.textContent = data.model_predictions.random_forest;
            gbPrediction.textContent = data.model_predictions.gradient_boosting;
            lrPrediction.textContent = data.model_predictions.logistic_regression;
            
            // Display genre icon if available
            if (genreIconElement) {
                const genre = data.ensemble_prediction.toLowerCase();
                genreIconElement.innerHTML = `<span style="font-size: 3rem;">${genreIcons[genre] || '🎵'}</span>`;
            }
            
            // Create or update chart
            updateChart(data.ensemble_probabilities, data.ensemble_prediction);
            
            // Hide loading, show results
            if (loadingCard) loadingCard.style.display = 'none';
            resultSection.style.display = 'block';
            
            // Scroll to results
            resultSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            console.error('Error:', error);
            
            // Hide loading
            if (loadingCard) loadingCard.style.display = 'none';
            
            // Show error message
            if (errorCard) {
                document.getElementById('error-message').textContent = error.message || 'An error occurred while classifying the lyrics.';
                errorCard.style.display = 'block';
            } else {
                // Otherwise use alert
                alert('Error classifying lyrics: ' + error.message);
            }
        });
        
    }

    // Function to simulate API response for testing frontend (remove in production)
    function simulateClassification(lyrics) {
        return new Promise((resolve) => {
            setTimeout(() => {
                // Extract some keywords to simulate different predictions based on input
                const lowerLyrics = lyrics.toLowerCase();
                
                // Determine genre based on keywords in the sample lyrics
                let mainGenre = 'pop'; // Default genre
                
                if (lowerLyrics.includes('whiskey') || lowerLyrics.includes('country roads') || lowerLyrics.includes('pickup truck')) {
                    mainGenre = 'country';
                } else if (lowerLyrics.includes('rhythm') || lowerLyrics.includes('dance') || lowerLyrics.includes('melody')) {
                    mainGenre = 'pop';
                } else if (lowerLyrics.includes('rhymes') || lowerLyrics.includes('hustling') || lowerLyrics.includes('streets')) {
                    mainGenre = 'hip hop';
                } else if (lowerLyrics.includes('saxophone') || lowerLyrics.includes('smooth') || lowerLyrics.includes('piano')) {
                    mainGenre = 'jazz';
                } else if (lowerLyrics.includes('blues') || lowerLyrics.includes('crying') || lowerLyrics.includes('twelve bars')) {
                    mainGenre = 'blues';
                } else if (lowerLyrics.includes('guitar solos') || lowerLyrics.includes('rock and roll') || lowerLyrics.includes('headbang')) {
                    mainGenre = 'rock';
                }
                
                // Generate random probabilities
                const genres = ['rock', 'pop', 'hip hop', 'country', 'jazz', 'blues', 'reggae'];
                const probabilities = {};
                
                // Make selected genre have highest probability
                let remainingProbability = 0.3; // 70% for main genre, 30% distributed among others
                
                genres.forEach(genre => {
                    if (genre === mainGenre) {
                        probabilities[genre] = 0.7;
                    } else {
                        const randomProb = Math.random() * remainingProbability / (genres.length - 1);
                        probabilities[genre] = randomProb;
                        remainingProbability -= randomProb;
                    }
                });
                
                // Ensure probabilities sum to 1
                const totalProb = Object.values(probabilities).reduce((sum, prob) => sum + prob, 0);
                Object.keys(probabilities).forEach(genre => {
                    probabilities[genre] = probabilities[genre] / totalProb;
                });
                
                // Sort probabilities for better visualization (highest first)
                const sortedProbabilities = {};
                Object.entries(probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .forEach(([genre, prob]) => {
                        sortedProbabilities[genre] = prob;
                    });
                
                // Random predictions for individual models
                let otherGenres = genres.filter(g => g !== mainGenre);
                const randomForestPrediction = Math.random() < 0.8 ? mainGenre : 
                    otherGenres[Math.floor(Math.random() * otherGenres.length)];
                
                const gradientBoostingPrediction = Math.random() < 0.7 ? mainGenre : 
                    otherGenres[Math.floor(Math.random() * otherGenres.length)];
                
                const logisticRegressionPrediction = Math.random() < 0.6 ? mainGenre : 
                    otherGenres[Math.floor(Math.random() * otherGenres.length)];
                
                resolve({
                    ensemble_prediction: mainGenre,
                    ensemble_probabilities: sortedProbabilities,
                    model_predictions: {
                        random_forest: randomForestPrediction,
                        gradient_boosting: gradientBoostingPrediction,
                        logistic_regression: logisticRegressionPrediction
                    }
                });
            }, 1000); // Simulate 1 second processing time
        });
    }

    // Function to update chart
    function updateChart(probabilities, predictedGenre) {
        // Prepare data for chart
        const labels = Object.keys(probabilities);
        const data = Object.values(probabilities);
        
        // Set background colors and create datasets with highlight for predicted genre
        const backgroundColors = [];
        const borderColors = [];
        const borderWidths = [];
        const hoverBackgroundColors = [];
        
        for (let i = 0; i < labels.length; i++) {
            const genre = labels[i].toLowerCase();
            const color = chartColors[genre] || '#777777';
            const isHighlighted = (labels[i].toLowerCase() === predictedGenre.toLowerCase());
            
            backgroundColors.push(isHighlighted ? color : color + 'CC'); // Make non-highlighted genres slightly transparent
            borderColors.push(isHighlighted ? '#FFFFFF' : 'rgba(255, 255, 255, 0.5)');
            borderWidths.push(isHighlighted ? 2 : 1);
            hoverBackgroundColors.push(color);
        }
        
        // Chart configuration
        const config = {
            type: currentChartType,
            data: {
                labels: labels,
                datasets: [{
                    label: 'Genre Probability',
                    data: data,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: borderWidths,
                    hoverBackgroundColor: hoverBackgroundColors
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: currentChartType === 'pie' ? 'right' : 'top',
                        labels: {
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                const percentage = (value * 100).toFixed(1) + '%';
                                const label = context.label;
                                let displayText = label + ': ' + percentage;
                                
                                // Add indicator for predicted genre
                                if (label.toLowerCase() === predictedGenre.toLowerCase()) {
                                    displayText += ' (Predicted)';
                                }
                                
                                return displayText;
                            },
                            title: function(context) {
                                return context[0].label;
                            }
                        }
                    }
                }
            }
        };
        
        // For radar chart, add specific options
        if (currentChartType === 'radar') {
            config.options.scales = {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            };
        }
        
        // For bar chart, add specific options
        if (currentChartType === 'bar') {
            config.options.indexAxis = 'y'; // Horizontal bar chart
            config.options.scales = {
                x: {
                    beginAtZero: true,
                    suggestedMax: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            };
        }

        // Add animations
        config.options.animation = {
            duration: 1000,
            easing: 'easeOutQuart'
        };
        
        // Destroy previous chart if it exists
        if (genreChart) {
            genreChart.destroy();
        }
        
        // Create new chart
        const ctx = document.getElementById('pie-chart').getContext('2d');
        if (ctx) {
            genreChart = new Chart(ctx, config);
        } else {
            console.error("Canvas context for chart not found");
        }
    }
});