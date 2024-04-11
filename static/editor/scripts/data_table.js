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
        if (src == 'edit') {
            save_table_data();
        }
    });
    hot.addHook('afterRedo', function(changes, src) {
        save_table_data();
    });
    hot.addHook('afterUndo', function(changes, src) {
        save_table_data();
    });
    hot.addHook('afterPaste', function(changes, src) {
        save_table_data();
    });

    function save_table_data()
    {
        var jsonData = {data: hot.getData()};
        save_data1(jsonData);
    }
    function save_data1(jsonData) {
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



    const new_output_btn = document.getElementById("new_output")

    new_output_btn.addEventListener('click', function(event){
        var jsonData = {data: hot.getData()};
        get_sample_names(jsonData)
    });
    function get_sample_names(jsonData) {
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
            // Extract sample names from the response data
            var sampleNames = data.sample_names;
            displaySampleNames(sampleNames);
        })
        .catch(error => {
            console.error('Error fetching sample names:', error);
        });
    }

    function displaySampleNames(sampleNames) {
    // Get the container for the sample list
    var sampleList = document.getElementById("sample_list");

    // Find the table within the container
    var table = sampleList.querySelector(".table_list");

    // Clear any existing content in the table body
    var tbody = table.querySelector("tbody");
    tbody.innerHTML = "";

    // Iterate over each sample name and create table rows
    sampleNames.forEach(function(sampleName, index) {
        // For the first sample name, create table header cells
        if (index === 0) {
            var trHeader = document.createElement("tr");
            var thSampleName = document.createElement("th");
            thSampleName.textContent = "Sample Name:";
            var thActive = document.createElement("th");
            thActive.textContent = "Active:";
            trHeader.appendChild(thSampleName);
            trHeader.appendChild(thActive);
            // Append the table header row to the table body
            tbody.appendChild(trHeader);
        }
        // Create table row and table data elements
        var tr = document.createElement("tr");
        tr.setAttribute("class", "table_list");
        var tdSampleName = document.createElement("td");
        var tdCheckbox = document.createElement("td");

        // Create label and checkbox for the sample
        var label = document.createElement("label");
        label.setAttribute("for", sampleName);
        label.textContent = sampleName;
        var checkbox = document.createElement("input");
        checkbox.setAttribute("type", "checkbox");
        checkbox.setAttribute("class", "sample-checkbox");
        checkbox.setAttribute("id", sampleName);
        checkbox.setAttribute("name", sampleName);
        checkbox.setAttribute("value", sampleName);
        checkbox.checked = true; // Assuming all samples are initially checked

        // Append label and checkbox to the table data elements
        tdSampleName.appendChild(label);
        tdCheckbox.appendChild(checkbox);

        // Append table data elements to the table row
        tr.appendChild(tdSampleName);
        tr.appendChild(tdCheckbox);

        // Append table row to the table body
        tbody.appendChild(tr);
        });
    }



});

