<h1 style="text-align: center">dzToolBox</h1>
<h3 style="text-align: center">Web-based tools for data analysis and interpretation for detrital zircon</h3>
<h5 style="text-align: center">Developed by Ryan Nielsen in conjunction with Kurt Sundell</h5>
<br />
<br />
<h3 style="text-align: center">How to navigate this code:</h3>
<h5>Python Code:</h5>
<p>The main app routes are found in the "apps" folder. 
They handle all requests from the frontend to the server 
and choose what information to send back to the front end. <br />
The "utils" folder stores libraries that analyze DZ data and 
output graphs, tables, stats, etc. These are called by the
functions in the "apps" folder. <br />
The "server" folder contains libraries that the "apps" folder 
calls to interact with the server. These handle the
database of users and projects, as well as cleaning up the
temporary files and data that are no longer useful to the
apps. <br />
The "instance" folder contains the database. It is created
and maintained via python code in "database.py" in the "server"
folder.</p>
<br />
<h5>HTML, JS, and CSS:</h5>
<p>Most of the code that handles displaying the apps to the
screen is found in the "templates" folder. For each app, there
is a directory in the templates folder that contains HTML files
to display information in an (hopefully) aesthetic way which 
simplifies managing samples and facilitates making graphs and 
tables. There is also a "global" folder of partial HTML files
that are reused across multiple apps.<br />
The "static" folder is laid out like the "templates" folder,
containing a directory for each app, as well as a "global"
directory. In each app directory there are (usually) two
folders: a "styles" folder and a "scripts" folder. In the 
"styles" folder are css files which control how the app looks.
In the "scripts" folder are javascript files that control
how the app behaves. All the functions that control the app,
that don't need to be done by the server, are included here.
</p>

<h5>Todo:</h5>
<p>Implement full dzMix functionality. Still need to 
develop KS and Kuiper tests instead of just R^2. Implement deleting individual outputs.</p>

<p>Finish writing the following files:</p>
<ul>
    <li style="color:cyan;">/apps</li>
    <ul>
        <li>editor.py</li>
    </ul>
    <li style="color:cyan;">/utils</li>
    <ul>
        <li>unmix.py</li>
    </ul>
</ul>
