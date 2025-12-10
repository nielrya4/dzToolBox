// Belt DB Geological Map JavaScript

// Global variables
var map = null;
var allMarkers = [];
var visibleMarkers = [];
var sampleData = {};
var markersLayer = null;
var currentSampleData = null;
var zirconTable = null;
var sampleMarkers = []; // Will be populated by the template

// Map belt group to CSS class name
var beltGroupToClass = {
    "Pre-Belt": "marker-pre-belt",
    "Lower Belt": "marker-lower-belt", 
    "Ravalli": "marker-ravalli",
    "Piegan": "marker-piegan",
    "Missoula": "marker-missoula",
    "Lemhi Subbasin": "marker-lemhi",
    "Post-Belt": "marker-post-belt"
};

// Function to initialize sample data
function initializeSampleData(markers) {
    sampleMarkers = markers;
    // Create sample data lookup
    sampleMarkers.forEach(function(marker) {
        sampleData[marker.name] = marker;
    });
    // Populate sample names for autocomplete
    allSampleNames = markers.map(function(marker) {
        return marker.name;
    }).sort();
}

// Function to apply CSS class to marker
function applyMarkerColor(marker, sampleName) {
    var sample = sampleData[sampleName];
    var cssClass = beltGroupToClass[sample.belt_gp_correlative] || "marker-unknown";
    
    // Wait for marker to be rendered then apply class
    setTimeout(function() {
        if (marker._icon) {
            // Remove any existing marker classes
            marker._icon.classList.remove('marker-pre-belt', 'marker-lower-belt', 'marker-ravalli', 
                                         'marker-piegan', 'marker-missoula', 'marker-lemhi', 
                                         'marker-post-belt', 'marker-unknown');
            // Apply the correct class
            marker._icon.classList.add(cssClass);
        }
    }, 10);
}

// Initialize map
function initializeMap() {
    // Initialize the map
    map = L.map('map').setView([46, -113], 6);
    
    // Add tile layers
    var openStreetMap = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    });
    
    var esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Esri',
        maxZoom: 19
    });
    
    // Add default layer
    esriSatellite.addTo(map);
    
    // Create layer control
    var baseLayers = {
        "OpenStreetMap": openStreetMap,
        "Satellite": esriSatellite
    };
    L.control.layers(baseLayers).addTo(map);
    
    // Add center/home control
    L.control.scale().addTo(map);
    
    // Add custom center button
    var centerControl = L.Control.extend({
        options: {
            position: 'topleft'
        },
        onAdd: function (map) {
            var container = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-custom');
            container.style.backgroundColor = 'white';
            container.style.width = '30px';
            container.style.height = '30px';
            container.style.cursor = 'pointer';
            container.innerHTML = '⊕'; // crosshairs symbol
            container.style.fontSize = '24px';
            container.style.fontWeight='Bold';
            container.style.textAlign = 'center';
            container.style.lineHeight = '30px';
            container.style.border = '2px solid rgba(0,0,0,0.2)';
            container.style.borderRadius = '4px';
            container.title = 'Reset map view';
            
            // Add hover effects
            container.onmouseover = function() {
                container.style.backgroundColor = '#f4f4f4';
            };
            container.onmouseout = function() {
                container.style.backgroundColor = 'white';
            };
            
            container.onclick = function(){
                resetMapView();
            }
            return container;
        }
    });
    map.addControl(new centerControl());
    
    // Create markers layer group
    markersLayer = L.layerGroup().addTo(map);
    
    // Add markers
    addMarkers();
    
    // Update statistics
    updateStatistics();
    
    // Add window resize handler
    window.addEventListener('resize', function() {
        setTimeout(function() {
            if (map) {
                map.invalidateSize();
            }
        }, 100);
    });
}

// Add markers to map
function addMarkers() {
    sampleMarkers.forEach(function(markerData) {
        var marker = L.marker([markerData.lat, markerData.lng], {
            title: 'Sample: ' + markerData.name
        });
        
        marker.on('click', function() {
            updateInfoPanel(markerData.name);
        });
        
        markersLayer.addLayer(marker);
        
        // Apply custom color class to marker icon
        applyMarkerColor(marker, markerData.name);
        
        allMarkers.push(marker);
        visibleMarkers.push(marker);
    });
}

// Function to update info panel
function updateInfoPanel(sampleName) {
    var sample = sampleData[sampleName];
    if (sample) {
        console.log('Updating info panel for sample:', sampleName, sample);
        
        // Update the currently selected sample and apply styling
        currentlySelectedSample = sampleName;
        applySelectedSampleStyling();
        
        document.getElementById('sampleInfo').innerHTML = `
            <h3>${sample.name}</h3>
            <p><strong>Formation:</strong> ${sample.formation || 'Unknown'}</p>
            <p><strong>Belt Group:</strong> ${sample.belt_gp_correlative || 'Unknown'}</p>
            <p><strong>Region:</strong> ${sample.region || 'Unknown'}</p>
            <p><strong>Reference:</strong> ${sample.reference || 'Unknown'}</p>
            <p><strong>Coordinates:</strong> ${(sample.lat || 0).toFixed(4)}, ${(sample.lng || 0).toFixed(4)}</p>
            <p><strong>Max Depositional Age:</strong> ${(sample.mda_age || 0).toFixed(1)} ± ${(sample.mda_uncertainty || 0).toFixed(1)} Ma</p>
            <p><strong>Zircon Grains:</strong> ${(sample.zircon_data || []).length} grains</p>
        `;
        
        // Show KDE graph if available
        if (sample.kde_graph) {
            document.getElementById('kdeGraphContainer').style.display = 'block';
            document.getElementById('kdeGraph').src = sample.kde_graph;
        } else {
            document.getElementById('kdeGraphContainer').style.display = 'none';
        }
        
        // Show zircon data table
        var zirconData = sample.zircon_data || [];
        if (zirconData.length > 0) {
            currentSampleData = sample;
            document.getElementById('tableContainer').style.display = 'block';
            document.getElementById('tableTitle').textContent = `Zircon Age Data - ${sample.name || 'Unknown'}`;
            
            // Prepare data for Handsontable  
            var tableData = [[sample.name, 'Uncertainty (±1σ)']].concat(
                zirconData.map(function(grain) {
                    return [grain.age, grain.uncertainty];
                })
            );
            
            // Destroy existing table if it exists
            if (zirconTable) {
                zirconTable.destroy();
            }
            
            // Create new Handsontable
            var container = document.getElementById('zirconTable');
            zirconTable = new Handsontable(container, {
                data: tableData,
                colHeaders: false,
                columns: [
                    { type: 'numeric', readOnly: true, numericFormat: { pattern: '0.0' } },
                    { type: 'numeric', readOnly: true, numericFormat: { pattern: '0.0' } }
                ],
                rowHeaders: false,
                width: '100%',
                height: 250,
                licenseKey: 'non-commercial-and-evaluation',
                stretchH: 'all',
                autoColumnSize: true,
                manualColumnResize: true,
                contextMenu: ['copy'],
                filters: true,
                dropdownMenu: true,
                copyPaste: true
            });
        } else {
            document.getElementById('tableContainer').style.display = 'none';
        }
    } else {
        console.error('Sample not found:', sampleName, 'Available samples:', Object.keys(sampleData).slice(0, 5));
        
        // Clear the currently selected sample
        currentlySelectedSample = null;
        applySelectedSampleStyling();
        
        document.getElementById('sampleInfo').innerHTML = `
            <h3>Sample Not Found</h3>
            <p>Sample "${sampleName}" data could not be loaded.</p>
        `;
        document.getElementById('tableContainer').style.display = 'none';
    }
}

// Action button functions
function resetMapView() {
    if (map) {
        map.setView([46, -113], 6);
    }
}

function toggleAllMarkers() {
    if (!markersLayer) return;
    
    if (map.hasLayer(markersLayer)) {
        map.removeLayer(markersLayer);
        visibleMarkers = [];
    } else {
        map.addLayer(markersLayer);
        visibleMarkers = allMarkers.slice();
    }
    updateStatistics();
}

function fitMapToBounds() {
    if (!markersLayer || allMarkers.length === 0) return;
    
    var group = L.featureGroup(visibleMarkers);
    map.fitBounds(group.getBounds(), {padding: [20, 20]});
}

// Unified filtering function that applies ALL active filters
function applyAllFilters(preserveSelection) {
    var selectedGroup = document.getElementById('beltGroupFilter').value;
    var minAge = parseFloat(document.getElementById('minAge').value);
    var maxAge = parseFloat(document.getElementById('maxAge').value);
    var searchType = document.getElementById('ageSearchType').value;
    var searchTerm = document.getElementById('sampleSearch').value.toLowerCase().trim();
    
    // Clear selection unless explicitly preserving it
    if (!preserveSelection) {
        currentlySelectedSample = null;
        // Clear info panel
        document.getElementById('sampleInfo').innerHTML = '<p class="no-selection">Click on a sample marker to view details</p>';
        document.getElementById('kdeGraphContainer').style.display = 'none';
        document.getElementById('tableContainer').style.display = 'none';
    }
    
    // Reset visible markers array
    visibleMarkers = [];
    
    // Filter markers without removing/adding to preserve event handlers
    allMarkers.forEach(function(marker) {
        var sampleName = marker.options.title.replace('Sample: ', '');
        var sample = sampleData[sampleName];
        var includeMarker = true;
        
        // Filter by Belt Group (if selected)
        if (selectedGroup && sample.belt_gp_correlative !== selectedGroup) {
            includeMarker = false;
        }
        
        // Filter by Age Range (if either min or max age is specified)
        if (includeMarker && (!isNaN(minAge) || !isNaN(maxAge))) {
            var effectiveMinAge = isNaN(minAge) ? 0 : minAge;
            var effectiveMaxAge = isNaN(maxAge) ? 4500 : maxAge;
            
            if (searchType === 'mda') {
                // Filter by Maximum Depositional Age
                if (sample.mda_age < effectiveMinAge || sample.mda_age > effectiveMaxAge) {
                    includeMarker = false;
                }
            } else if (searchType === 'grain') {
                // Filter by grain ages - include sample if any grain falls within range
                if (sample.zircon_data && sample.zircon_data.length > 0) {
                    var hasGrainInRange = sample.zircon_data.some(function(grain) {
                        return grain.age >= effectiveMinAge && grain.age <= effectiveMaxAge;
                    });
                    if (!hasGrainInRange) {
                        includeMarker = false;
                    }
                } else {
                    includeMarker = false;
                }
            }
        }
        
        // Filter by Sample Name (if search term is provided)
        if (includeMarker && searchTerm && !sampleName.toLowerCase().includes(searchTerm)) {
            includeMarker = false;
        }
        
        // Show or hide marker based on filter results
        if (includeMarker) {
            if (!markersLayer.hasLayer(marker)) {
                markersLayer.addLayer(marker);
            }
            // Ensure marker color is applied
            applyMarkerColor(marker, sampleName);
            visibleMarkers.push(marker);
        } else {
            if (markersLayer.hasLayer(marker)) {
                markersLayer.removeLayer(marker);
            }
        }
    });
    
    // Apply selected sample styling
    applySelectedSampleStyling();
    
    updateStatistics();
}

// Track currently selected sample and ring marker
var currentlySelectedSample = null;
var selectionRingMarker = null;

// Function to apply red ring styling to the currently selected sample
function applySelectedSampleStyling() {
    // Remove existing ring marker if it exists
    if (selectionRingMarker) {
        map.removeLayer(selectionRingMarker);
        selectionRingMarker = null;
    }
    
    // If there's a currently selected sample, add the red ring to it
    if (currentlySelectedSample) {
        var selectedMarker = allMarkers.find(function(marker) {
            var sampleName = marker.options.title.replace('Sample: ', '');
            return sampleName === currentlySelectedSample;
        });
        
        if (selectedMarker) {
            // Get the position of the selected marker
            var lat = selectedMarker.getLatLng().lat;
            var lng = selectedMarker.getLatLng().lng;
            
            // Create a custom ring icon
            var ringIcon = L.divIcon({
                className: 'selection-ring-marker',
                html: '<div class="selection-ring"></div>',
                iconSize: [24, 24],
                iconAnchor: [17.5, 20] // Center horizontally, shift up 15px from center (17.5 + 15)
            });
            
            // Create the ring marker at the same position
            selectionRingMarker = L.marker([lat, lng], {
                icon: ringIcon,
                interactive: false, // Don't interfere with clicks
                zIndexOffset: -1000 // Put it behind other markers
            }).addTo(map);
        }
    }
}

// Individual filter functions now call the unified filter
function filterByBeltGroup() {
    applyAllFilters();
}

function filterByAge() {
    applyAllFilters();
}

var currentSuggestionIndex = -1;
var allSampleNames = [];

function handleSampleSearch() {
    var searchTerm = document.getElementById('sampleSearch').value.toLowerCase();
    var suggestionsContainer = document.getElementById('sampleSuggestions');
    
    if (searchTerm.length === 0) {
        // Hide suggestions and apply all other filters
        suggestionsContainer.style.display = 'none';
        applyAllFilters();
        return;
    }
    
    // Filter sample names based on search term
    var matchingSamples = allSampleNames.filter(function(name) {
        return name.toLowerCase().includes(searchTerm);
    });
    
    if (matchingSamples.length > 0) {
        // Show suggestions dropdown
        showSuggestions(matchingSamples, searchTerm);
        
        // Apply all filters including search term
        applyAllFilters();
    } else {
        // Hide suggestions if no matches
        suggestionsContainer.style.display = 'none';
        // Apply all filters (will show no results due to search term)
        applyAllFilters();
    }
}

function showSuggestions(suggestions, searchTerm) {
    var suggestionsContainer = document.getElementById('sampleSuggestions');
    suggestionsContainer.innerHTML = '';
    
    // Limit to first 10 suggestions
    suggestions.slice(0, 10).forEach(function(suggestion, index) {
        var suggestionItem = document.createElement('div');
        suggestionItem.className = 'suggestion-item';
        suggestionItem.textContent = suggestion;
        
        suggestionItem.onclick = function() {
            selectSample(suggestion);
        };
        
        suggestionsContainer.appendChild(suggestionItem);
    });
    
    suggestionsContainer.style.display = 'block';
    currentSuggestionIndex = -1;
}

function selectSample(sampleName) {
    document.getElementById('sampleSearch').value = sampleName;
    document.getElementById('sampleSuggestions').style.display = 'none';
    
    // Apply all filters with the selected sample name, preserving selection
    applyAllFilters(true);
    
    // Center map on the selected sample and show details
    var sample = sampleData[sampleName];
    if (sample) {
        map.setView([sample.lat, sample.lng], 10);
        updateInfoPanel(sampleName);
    }
}


function showAllMarkers() {
    // Clear all filter inputs
    document.getElementById('beltGroupFilter').value = '';
    document.getElementById('ageSearchType').value = 'mda';
    document.getElementById('minAge').value = '';
    document.getElementById('maxAge').value = '';
    document.getElementById('sampleSearch').value = '';
    document.getElementById('sampleSuggestions').style.display = 'none';
    
    // Apply filters (which will show all markers since all filters are cleared)
    applyAllFilters();
}

function clearFilters() {
    document.getElementById('beltGroupFilter').value = '';
    document.getElementById('ageSearchType').value = 'mda';
    document.getElementById('minAge').value = '';
    document.getElementById('maxAge').value = '';
    document.getElementById('sampleSearch').value = '';
    document.getElementById('sampleSuggestions').style.display = 'none';
    
    // Apply all filters (which will show all markers since filters are cleared)
    applyAllFilters();
}

function updateStatistics() {
    document.getElementById('visibleSamples').textContent = visibleMarkers.length;
    
    if (visibleMarkers.length > 0) {
        var ages = visibleMarkers.map(function(marker) {
            var sampleName = marker.options.title.replace('Sample: ', '');
            return sampleData[sampleName].mda_age;
        });
        
        var minAge = Math.min.apply(Math, ages);
        var maxAge = Math.max.apply(Math, ages);
        document.getElementById('ageRange').textContent = minAge.toFixed(1) + '-' + maxAge.toFixed(1) + ' Ma';
    } else {
        document.getElementById('ageRange').textContent = 'No data';
    }
}

function exportVisibleData() {
    var csvContent = 'Sample Name,Formation,Belt Group,Region,Reference,Latitude,Longitude,MDA Age,MDA Uncertainty\\n';
    
    visibleMarkers.forEach(function(marker) {
        var sampleName = marker.options.title.replace('Sample: ', '');
        var sample = sampleData[sampleName];
        
        csvContent += [
            sample.name,
            sample.formation,
            sample.belt_gp_correlative,
            sample.region,
            sample.reference,
            sample.lat,
            sample.lng,
            sample.mda_age,
            sample.mda_uncertainty
        ].join(',') + '\\n';
    });
    
    downloadCSV(csvContent, 'visible_samples.csv');
}

function exportAllData() {
    var csvContent = 'Sample Name,Formation,Belt Group,Region,Reference,Latitude,Longitude,MDA Age,MDA Uncertainty\\n';
    
    Object.values(sampleData).forEach(function(sample) {
        csvContent += [
            sample.name,
            sample.formation,
            sample.belt_gp_correlative,
            sample.region,
            sample.reference,
            sample.lat,
            sample.lng,
            sample.mda_age,
            sample.mda_uncertainty
        ].join(',') + '\\n';
    });
    
    downloadCSV(csvContent, 'all_samples.csv');
}

function exportMapImage() {
    alert('Map image export functionality would require additional libraries. Consider using browser print/screenshot for now.');
}

function downloadCSV(content, filename) {
    var blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    var link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function exportSampleData() {
    if (!currentSampleData || !currentSampleData.zircon_data) {
        alert('No sample data to export');
        return;
    }
    
    var csvContent = 'Age (Ma),Uncertainty (±1σ)\\n';
    
    currentSampleData.zircon_data.forEach(function(grain) {
        csvContent += grain.age + ',' + grain.uncertainty + '\\n';
    });
    
    var filename = currentSampleData.name + '_zircon_data.csv';
    downloadCSV(csvContent, filename);
}

function exportRawDatabase() {
    // Create a link to download the raw database file
    var link = document.createElement('a');
    link.href = 'data/beltdb.xlsx';
    link.download = 'beltdb_raw_database.xlsx';
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function exportTabulatedData() {
    // Create a link to download the tabulated data file
    var link = document.createElement('a');
    link.href = 'data/beltdb_tabulated.xlsx';
    link.download = 'beltdb_tabulated_data.xlsx';
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function handleMultipleExports() {
    // Get selected export options from MultiSelect
    var selectedExports = [];
    
    // The MultiSelect library creates hidden inputs with name pattern like "multi-select-[id][]"
    // Let's find the correct selector by looking for the multi-select container
    var multiSelectElement = document.querySelector('.multi-select');
    if (multiSelectElement) {
        var hiddenInputs = multiSelectElement.querySelectorAll('input[type="hidden"]');
        hiddenInputs.forEach(function(input) {
            if (input.value) {
                selectedExports.push(input.value);
            }
        });
    }
    
    if (selectedExports.length === 0) {
        alert('Please select at least one export option');
        return;
    }
    
    // Execute selected exports with small delays to avoid browser blocking
    selectedExports.forEach(function(exportType, index) {
        setTimeout(function() {
            switch(exportType) {
                case 'raw-database':
                    exportRawDatabase();
                    break;
                case 'tabulated-data':
                    exportTabulatedData();
                    break;
                default:
                    console.warn('Unknown export type:', exportType);
            }
        }, index * 500); // 500ms delay between exports
    });
}

// Fetch sample data from JSON file
function fetchSampleData() {
    return fetch('/static/belt_db/data/markers.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            initializeSampleData(data);
            // Update total samples count
            document.getElementById('totalSamples').textContent = data.length;
            return data;
        })
        .catch(error => {
            console.error('Error fetching sample data:', error);
            alert('Error loading sample data. Please ensure the data file exists.');
            return [];
        });
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    fetchSampleData().then(() => {
        initializeMap();
        // Add keyboard navigation and click-outside handler for autocomplete
        addAutocompleteEventHandlers();
    });
});

// Add keyboard navigation and click-outside handler for autocomplete
function addAutocompleteEventHandlers() {
    var searchInput = document.getElementById('sampleSearch');
    var suggestionsContainer = document.getElementById('sampleSuggestions');
    
    // Keyboard navigation
    searchInput.addEventListener('keydown', function(e) {
        var suggestions = suggestionsContainer.querySelectorAll('.suggestion-item');
        var highlightedIndex = -1;
        
        // Find currently highlighted suggestion
        for (var i = 0; i < suggestions.length; i++) {
            if (suggestions[i].classList.contains('highlighted')) {
                highlightedIndex = i;
                break;
            }
        }
        
        switch(e.key) {
            case 'ArrowDown':
                e.preventDefault();
                if (suggestionsContainer.style.display !== 'none') {
                    // Remove current highlight
                    if (highlightedIndex >= 0) {
                        suggestions[highlightedIndex].classList.remove('highlighted');
                    }
                    // Move to next suggestion
                    var nextIndex = highlightedIndex + 1;
                    if (nextIndex >= suggestions.length) nextIndex = 0;
                    if (suggestions[nextIndex]) {
                        suggestions[nextIndex].classList.add('highlighted');
                    }
                }
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                if (suggestionsContainer.style.display !== 'none') {
                    // Remove current highlight
                    if (highlightedIndex >= 0) {
                        suggestions[highlightedIndex].classList.remove('highlighted');
                    }
                    // Move to previous suggestion
                    var prevIndex = highlightedIndex - 1;
                    if (prevIndex < 0) prevIndex = suggestions.length - 1;
                    if (suggestions[prevIndex]) {
                        suggestions[prevIndex].classList.add('highlighted');
                    }
                }
                break;
                
            case 'Enter':
                e.preventDefault();
                if (highlightedIndex >= 0 && suggestions[highlightedIndex]) {
                    var selectedText = suggestions[highlightedIndex].textContent;
                    selectSample(selectedText);
                }
                break;
                
            case 'Escape':
                suggestionsContainer.style.display = 'none';
                break;
        }
    });
    
    // Click outside to hide suggestions
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !suggestionsContainer.contains(e.target)) {
            suggestionsContainer.style.display = 'none';
        }
    });
    
    // Also add mouseenter/mouseleave for visual feedback
    suggestionsContainer.addEventListener('mouseenter', function(e) {
        if (e.target.classList.contains('suggestion-item')) {
            // Remove all highlights
            suggestionsContainer.querySelectorAll('.suggestion-item').forEach(function(item) {
                item.classList.remove('highlighted');
            });
            // Add highlight to hovered item
            e.target.classList.add('highlighted');
        }
    });
}