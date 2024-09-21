<h1 style="text-align: center">dzToolBox</h1>
<h3 style="text-align: center">Web-based tools for data analysis and interpretation for detrital zircon</h3>
<h5 style="text-align: center">Developed by Ryan Nielsen in conjunction with Kurt Sundell</h5>
<a href="https://www.dztoolbox.com">Try it out</a>
<h3>Features:</h3>
<p>(See <a href="https://www.dztoolbox.com/gsa_poster.pdf">my GSA poster</a> for more information!)</p>
<ul>
    <li>
        <h5>Stats</h5>
        <ul>
            <li>
                Graphs
                <ul>
                    <li>KDE (Kernel Density Estimate)</li>
                    <li>PDP (Probability Density Function)</li>
                    <li>CDF (Cumulative Distribution Function)</li>
                </ul>
            </li>
            <li>
                Matrices based on the following metrics:
                <ul>
                    <li>Similarity</li>
                    <li>Likeness</li>
                    <li>Cross-Correlation</li>
                    <li>KS Test</li>
                    <li>Kuiper Test</li>
                </ul>
            </li>
        </ul>
    </li>
    <li>
        <h5>MDS</h5>
        <p>MDS plots based on the following metrics:</p>
        <ul>
            <li>Similarity</li>
            <li>Likeness</li>
            <li>Cross-Correlation</li>
            <li>KS Test</li>
            <li>Kuiper Test</li>
        </ul>
    </li>
    <li>
        <h5>Unmix</h5>
        <ul>
            <li>
                <p>Unmixing using the metrics:</p>
                <ul>
                    <li>Cross-Correlation</li>
                    <li>KS Test</li>
                    <li>Kuiper Test</li>
                </ul>
            </li>
            <li>
                <p>Generate the outputs:</p>
                <ul>
                    <li>Relative Contribution Table</li>
                    <li>Relative Contribution Graph</li>
                    <li>Top Trials Graph</li>
                </ul>
            </li>
        </ul>
    </li>
</ul>
<h3>To Do:</h3>
<ul>
    <li>
        <h5>Project Settings</h5>
        <ul>
            <li>Font Size</li>
            <li>Graph Color Scheme</li>
            <li>X and Y Labels</li>
            <li>Editable Age Range</li>
            <li>Figure Dimensions</li>
        </ul>
    </li>
    <li>
        <h5>Project Browser</h5>
        <ul>
            <li>Rename Project</li>
        </ul>
    </li>
    <li>
        <h5>Editor</h5>
        <ul>
            <li>Add More Spreadsheets</li>
        </ul>
    </li>
    <li>
        <h5>Future Directions</h5>
        <ul>
            <li>MDA</li>
            <li>NMF</li>
            <li>Multivariate Data Interpretation and Visualization</li>
        </ul>
    </li>
</ul>
<br />
<p>Download this code by running:

```bash
git clone https://www.github.com/nielrya4/dztoolbox.git
```
And run it locally with the following:
```bash
cd dztoolbox
. venv/bin/activate
cd setup/linux
./setup.sh
./setup/linux/run_local.sh
```
</p>
<br />
<h3>How to navigate this code:</h3>
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


