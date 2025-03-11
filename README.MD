<main class="content">
    <section id="introduction">
        <br />
        <h2>dzToolBox Documentation</h2>
        <br />
        <h3>Introduction</h3>
        <p>
            dzToolBox is a web app that seeks to facilitate and increase access to methods and tools for comparing,
            visualizing, and interpreting detrital zircon age data. These methods and tools are modeled after
            existing Matlab versions such as DZstats, DZmix, DZmda, and so on. This documentation page will go
            through each part of the website and how to use it.
        </p>
    </section>
    <br />
    <section id="accounts">
        <h3>Accounts and Projects</h3>
        <p>
            dzToolBox allows people to create accounts and store their projects on our servers, free of charge.
            This is accomplished by compressing the data before storing it to keep our database small and fast.
            To use this software, you must either create an account or use a guest account. Keep in mind that
            all guest accounts are wiped from the database every day at midnight MST. If you would like your
            data to persist beyond that, please create a user account. Users and guests alike have the ability
            to make as many projects as they wish, with as much input and output data as they need. Once you
            are signed in with a user account, you will be taken to the <b>Projects</b> page where you can create,
            delete, or edit your projects. When you create a project, you have the option of loading a file
            (.xlsx, .xls, or .csv) to serve as your project data. This is optional, as project data can also
            be copied from another spreadsheet and pasted into the spreadsheet in the <b>Project Editor</b>.
            If you sign in with a guest account, you will be taken straight into
            the <b>Project Editor</b> with example data loaded for you. You can always access the <b>Projects</b>
            page by clicking the big zircon (the digital, blue megacryst) in the top left corner. Your projects
            save automatically when you make changes, provided you are connected to the internet. If you would
            like, you can find the link tod elete your account by clicking the <b>Account Settings</b> button on
            the <b>Projects</b> page.
        </p>
    </section>
    <br />
    <section id="data-formatting">
        <h3>Data Formatting</h3>
        <p>
            Before you jump into making any projects, you should make sure your samples are formatted correctly.
            Samples should be tabulated such that each sample takes up two consecutive columns in a spreadsheet.
            The left column begins with a sample name, under which are the grain ages. The second column can
            have either a title (like 1s, 2s, etc.), or the first cell can be blank. Under the first cell are
            the uncertainties associated with the grain ages. dzToolBox assumes that all listed uncertainties
            are at 1s (1 * standard deviation). If your uncertainties are at 2s, you should make them 1s.
            You can see a great example if you log into a guest account. There can be as many samples in a
            spreadsheet as needed, but be aware that the more samples you try to process at once, the slower
            that dzToolBox will operate. 10 samples seems to be a good limit, and most projects do not need to
            process many more than that at a time.
        </p>
    </section>
    <br />
    <section id="spreadsheet">
        <h3>Spreadsheet</h3>
        <p>
            The first page of the <b>Project Editor</b> is a simple spreadsheet editor. This is primarily used
            to change the names of your samples, but there are also some other features. If you would like to
            import more data from somewhere on your computer, use the <b>Import Data</b> button to do so. It
            will prompt you to select a file (.xlsx or .csv), then populate a data table with the contents of
            that file. Simply copy and paste the portions you would like to add. This is especially useful for
            analysing data that was sampled in similar areas using a mixture model (<b>Unmix</b>). The
            <b>Databases</b> button is similar, but it allows you to pull data from the available, published
            zircon databases. Currently, these include the Global Igneous and Global Detrital Databases compiled
            by Puetz and Condie in 2019, as well as the Belt Database compiled by Kurt Sundell in 2025. The
            <b>Export Spreadsheet</b> button facilitates saving your spreadsheet as a .XLSX, .XLS, .CSV, or
            .json file. Every time you edit the spreadsheet, it saves to the database.
        </p>
    </section>
    <br />
    <section id="outputs">
        <h3>Outputs</h3>
        <p>
            The outputs page of the <b>Project Editor</b> is probably where you will spend the most time while
            using dzToolBox. There, you can create, download, and/or delete graphs and tables. Use the buttons
            on the left side of the toolbar to launch the prompts to make outputs. On the right side of the
            toolbar, you can access the <b>Project Settings</b> or clear all the outputs. Clearing the outputs
            is something that cannot be undone, so tread lightly.
        </p>
    </section>
    <br />
    <section id="project-settings">
        <h3>Project Settings</h3>
        <p>
            In the <b>Project Settings</b>, you can change some environmental variables that determine how tests
            are run and how graphs are made. For example, you can change the KDE bandwidth, the age range to
            plot, or the colormap of the graph lines and points. There are many ways to tweak the environment to
            suit your needs. Settings are saved within the project as soon as you click the <b>Update
            Settings</b> button, so they will be persistent between sessions and will only apply to the project
            in which you set them.
        </p>
    </section>
    <br />
    <section id="mda">
        <h3>Maximum Depositional Age (MDA)</h3>
        <p>
            With the <b>MDA</b> dialog, you can find the maximum depositional age of your samples.
            MDA shows when the youngest "part" of a rock formed. Zircons are to rocks like chocolate chips are
            to cookies. The cookie itself cannot be older than the youngest chocolate chips. Thus, a sedimentary
            rock cannot be older than the youngest zircons. MDA is found using several metrics proposed in the
            literature over the years. The ones included in dzToolBox are: youngest single grain, youngest grain
            cluster at 1s, youngest grain cluster at 2s, youngest 3 zircons, youngest 3 overlapping zircons,
            youngest graphical peak, youngest gaussian fit, youngest statistical population, and the Tau method.
            These MDA functions only analyze one sample at a time.
        </p>
    </section>
    <br />
    <section id="distributions">
        <h3>Distributions and Matrices</h3>
        <p>
            With the <b>Distributions</b> dialog, you can make KDE (Kernel Density Estimate), PDP (Probability
            Density Plot), and CDF (Cumulative Distribution Function) graphs of your samples. Remember that the
            KDE bandwidth, colormap, and more can be changed in <b>Project Settings</b>, which can affect how
            your samples are graphed. The graphs put out by these functions can be downloaded in several image
            formats.
            <br />
            With the <b>Matrices</b> dialog, you can test how your samples relate to each other with a series of
            metrics. These include r^2, similarity, KS, Kuiper, likeness, and the inverses of the
            aforementioned. These matrix outputs can be downloaded as .XLSX, .XLS, and .CSV files.
        </p>
    </section>
    <br />
    <section id="mds">
        <h3>Multidimensional Scaling Models (MDS)</h3>
        <p>
            An MDS model is a graphical representation of a comparison matrix. It shows each sample as a point.
            A sample's nearest neighbor is the one that it is the most similar to. The metrics that are used
            when creating an MDS graph are the very same that are used to create the matrices described in the
            previous section.
        </p>
    </section>
    <br />
    <section id="unmixing">
        <h3>Unmixing Models</h3>
        <p>
            dzToolBox is the first web-based geochronology software to provide an environment for creating
            top-down mixture models. In this model, a sedimentary sink sample is compared with at least 2
            igneous source samples. The idea is that the sink sample only contains zircons from the source
            samples. Thus, for the model to give valid results, all known sources must be included in the
            spreadsheet. The model goes through 10,000 iterations where a set of random weights are generated
            for each of the sources. The composite of randomly weighted sources is compared against the sink
            sample, either with an r^2, KS, or Kuiper test. The average test results of the top 10 trials are
            used to determine the percent contributions of the igneous sources. There are 3 choices of outputs
            for unmixing using dzToolBox: a Top Trials Graph, a Relative Contributions Graph, and a Relative
            Contributions Table. When you use the <b>Unmix</b> dialog, you have to have your data formatted
            a certain way. The sink sample must be the first (leftmost) sample in your spreadsheet. Then all the
            source samples can be in whatever order you want.
        </p>
    </section>
    <br />
    <section id="hafnium">
        <h3>Hafnium Plotter</h3>
        <p>
            Hafnium Plotter is a tool to visualize 2-dimensional data for samples that have both U-Pb and
            Hafnium data. Essentially, what is generated is a bivariate KDE plot. This can be visualized in 2
            ways: a heatmap and an interactive, 3D model. You can rotate and zoom the 3D model to show
            important features, and then click the small camera icon to save the image. The heatmap is similar
            to all other dzToolBox graphs in the way that it is downloaded.
        </p>
    </section>
    <br />
    <section id="future-steps">
        <h3>Future Steps</h3>
        <p>
            Aside from finding and fixing small errors in dzToolBox, there are some changes that will be made
            over time. These changes will first be manifest in the Python package behind all the functionality
            of this site: dz_lib. Once these changes and additions are implemented in the library,
            infrastructure that utilizes the new features will be added to dzToolBox. Some future steps include
            the ability of using metadata along with age and/or Hafnium data to create interactive web maps.
            The dz_lib library is already being used in one such context with the belt database. My plan is to
            make this functionality available to dzToolBox users by the end of 2025.
            <br />
            <br />
            I will also be developing a set of tools to analyze microplastics for my Master's Degree. These will
            doubtless end up somewhere on the web. This tools will mirror those that we use to interpret DZ
            data, and so I expect that there will be some new features that will benefit both the microplastics
            project and dzToolBox. I will link them here when I have them in a usable state.
        </p>
    </section>
    <br />
    <p>Thanks for using dzToolBox!</p>
    <br />
    <br />
    <h3>More Links:</h3>
    <p>
        Check out my <a href="https://www.dztoolbox.com/docs/gsa_poster.pdf" target="_blank">GSA poster</a> from Connects 2024!
    </p>
    <p>
        Visit my personal website, <a href="https://ryannielsen.com" target="_blank">ryannielsen.com</a>
    </p>
    <p>
        See what I've been up to on the native plants side of things at
        <a href="https://rewildidaho.org" target="_blank">rewildidaho.org</a>
    </p>
    <p>
        <a href="https://www.dztoolbox.com/docs/api" target="_blank">Guide</a> to using the dzToolBox API (Under Development)
    </p>
</main>