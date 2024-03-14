const new_project = document.getElementById("new_project")
const create_new_project = document.getElementById("create_new_project")
const project_name = document.getElementById("project_name")
const data_file = document.getElementById("data_file")


new_project.addEventListener('click',()=>{
    const docBox = new WinBox({
        title: 'Documentation',
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
        data_file.value = null;
        project_name.value = "";
        docBox.close();
    })
})

