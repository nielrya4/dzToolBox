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

    document.getElementById('save').addEventListener('click', function () {
        save_table_data();
    });

    document.getElementById('load').addEventListener('click', function () {
            document.getElementById('fileInput').click();
    });
    document.getElementById('fileInput').addEventListener('change', function (event) {
        var file = event.target.files[0];
        var reader = new FileReader();

        reader.onload = function (e) {
            var data = new Uint8Array(e.target.result);
            var workbook = XLSX.read(data, { type: 'array' });
            var sheetName = workbook.SheetNames[0];
            var worksheet = workbook.Sheets[sheetName];

            // Convert the entire range of cells to JSON format
            var excelData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

            // Determine the number of columns by finding the maximum row length
            var columnCount = Math.max(...excelData.map(row => row.length));

            // Fill any missing cells in each row
            for (var i = 0; i < excelData.length; i++) {
                var row = excelData[i];
                while (row.length < columnCount) {
                    row.push('');
                }
            }
            // Load Excel data into HandsOnTable
            hot.loadData(excelData);
            console.log('Data loaded from Excel:', excelData);
            //Save the file
            save_table_data();
        };
        reader.readAsArrayBuffer(file);
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
});

