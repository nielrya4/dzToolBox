const new_project = document.getElementById("new_project")
const create_new_project = document.getElementById("create_new_project")
const project_name = document.getElementById("project_name")
const data_file = document.getElementById("data_file")



new_project.addEventListener('click',()=>{
    data_file.value = null;
    project_name.value = "";
    const docBox = new WinBox({
        title: 'New Project',
        class: ["no-full"],
        width:'400px',
        height:'400px',
        background: '#003060',
        top:50,
        right:50,
        bottom:50,
        left:50,
        border: 2,
        mount: document.getElementById("content"),
        index: 1001
    })
    create_new_project.addEventListener('click', () =>{
        alert(data_file.value);
        docBox.close();
    })

    document.getElementById('data_file').addEventListener('change', function (event) {
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
            console.log('Data loaded from Excel:', excelData);

            // Save the file
            save_data(excelData);
        };

        reader.readAsArrayBuffer(file);
    });

    function save_data(jsonData) {
        sessionStorage.setItem("open_project", "0");

        fetch('/json/save/new_file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 'jsonData': jsonData })
        }).then(response => {
            if (response.ok) {
                console.log('Data saved successfully');
            } else {
                console.error('Error saving data');
            }
        }).catch(error => {
            console.error('Error saving data:', error);
        });
    }

})

