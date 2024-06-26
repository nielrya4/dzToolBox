document.addEventListener('DOMContentLoaded', function () {
    // Retrieve spreadsheet data from hidden HTML element
    var spreadsheetDataString = document.getElementById('spreadsheetData').value;
    console.log('Spreadsheet data string:', spreadsheetDataString);

    var spreadsheetData = [];
    if (spreadsheetDataString) {
        spreadsheetData = JSON.parse(spreadsheetDataString);
    }

    var container = document.getElementById('hot');
    var hot = new Handsontable(container, {
        data: spreadsheetData,
        licenseKey: 'non-commercial-and-evaluation',
        rowHeaders: true,
        colHeaders: true,
        contextMenu: true,
        renderAllRows: false,
        renderAllColumns: false,
        formulas: {
            engine: HyperFormula
        }
    });

    hot.addHook('afterChange', function(changes, src) {
        if (src === 'edit') {
            saveTableData();
        }
    });
    hot.addHook('afterRedo', saveTableData);
    hot.addHook('afterUndo', saveTableData);
    hot.addHook('afterPaste', saveTableData);

    function saveTableData() {
        var jsonData = { data: hot.getData() };
        saveData(jsonData);
    }

    function saveData(jsonData) {
        fetch('/json/save/spreadsheet', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 'jsonData': jsonData })
        }).then(response => {
            if (response.ok) {
                console.log('Data saved successfully!');
            } else {
                console.error('Error saving data');
            }
        }).catch(error => {
            console.error('Error saving data:', error);
        });
    }

    // Common event handler for buttons
    function handleButtonClick(event) {
        var jsonData = { data: hot.getData() };
        getSampleNames(jsonData);
    }

    // Add event listeners to buttons
    const buttons = ['new_output', 'new_mds', 'new_unmix'];
    buttons.forEach(buttonId => {
        const button = document.getElementById(buttonId);
        button.addEventListener('click', handleButtonClick);
    });

    function getSampleNames(jsonData) {
        fetch('/get_sample_names', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 'jsonData': jsonData })
        })
        .then(response => {
            if (response.ok) {
                return response.json(); // Parse the response body as JSON
            } else {
                throw new Error('Error fetching sample names');
            }
        })
        .then(data => {
            var sampleNames = data.sample_names;
            displaySampleNames(sampleNames);
        })
        .catch(error => {
            console.error('Error fetching sample names:', error);
        });
    }

    function displaySampleNames(sampleNames) {
        // Function to create table content
        function createTableContent(container) {
            container.innerHTML = "";

            var table = document.createElement("table");
            table.classList.add("table_list");
            var tbody = document.createElement("tbody");

            sampleNames.forEach(function(sampleName, index) {
                if (index === 0) {
                    var trHeader = document.createElement("tr");
                    var thSampleName = document.createElement("th");
                    thSampleName.textContent = "Sample Name:";
                    var thActive = document.createElement("th");
                    thActive.textContent = "Active:";
                    trHeader.appendChild(thSampleName);
                    trHeader.appendChild(thActive);
                    tbody.appendChild(trHeader);
                }

                var tr = document.createElement("tr");
                tr.setAttribute("class", "table_list");
                var tdSampleName = document.createElement("td");
                var tdCheckbox = document.createElement("td");
                var checkContainer = document.createElement("div");
                checkContainer.classList.add("form-check");
                checkContainer.classList.add("form-switch");

                var checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.classList.add("sample-checkbox");
                checkbox.classList.add("form-check-input");
                checkbox.value = sampleName;
                checkbox.name = sampleName;
                checkbox.id = sampleName;
                checkbox.checked = true;

                var label = document.createElement("label");
                label.setAttribute("for", sampleName);
                label.textContent = sampleName;
                label.classList.add("form-check-label");

                tdSampleName.appendChild(label);
                checkContainer.appendChild(checkbox);
                tdCheckbox.appendChild(checkContainer);
                tr.appendChild(tdSampleName);
                tr.appendChild(tdCheckbox);
                tbody.appendChild(tr);

                // Attach event listener to the row
                tr.addEventListener("click", function(event) {
                    if (event.target !== checkbox) {
                        checkbox.checked = !checkbox.checked;
                    }
                });

                // Attach event listener to the checkbox
                checkbox.addEventListener("click", function(event) {
                    event.stopPropagation(); // Prevent row click event
                });
            });

            table.appendChild(tbody);
            container.appendChild(table);
        }

        // Update each list with the created table content
        var sampleList = document.getElementById("sample_list");
        var mdsSampleList = document.getElementById("mds_sample_list");
        var unmixSampleList = document.getElementById("unmix_sample_list");

        // Clear the lists first to ensure no duplicates
        sampleList.innerHTML = "";
        mdsSampleList.innerHTML = "";
        unmixSampleList.innerHTML = "";

        // Create table content for each list
        createTableContent(sampleList);
        createTableContent(mdsSampleList);
        createTableContent(unmixSampleList);
    }
});
