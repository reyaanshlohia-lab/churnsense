# ChurnSense: the free website version

A website that predicts who is likely to **leave** or **join**, explains why, and draws every chart with its best
value marked ★. Visitors drag and drop their spreadsheets (or pick them with the file explorer), see the uploaded
files listed above the drop box, and type a request in plain English.

**Everything runs inside the visitor's browser.** There is no server running Python, so hosting is free with no card,
and spreadsheets never leave the visitor's computer. (The Python analysis code runs in the browser with Pyodide.)

## What is in this folder

There are **no sub-folders**: all files sit side by side, so they upload anywhere as ordinary files.

| Files | What they are |
|---|---|
| `index.html`, `styles.css`, `app.js` | The page: welcome (sign-in) screen, dashboard, files list, drag-and-drop box, prompt box, answer, and the sidebar pages. |
| `email-config.js` | Optional settings for sending result emails directly (see **Share results** below). |
| `worker.js` | Runs the Python analysis engine in the background, so the page never freezes. |
| `engine.py` | The analysis code (the same as `people_prediction_app.ipynb`, adapted for the browser). |
| `pyodide.*`, `python_stdlib.zip`, `pyodide-lock.json`, the `.whl` files, `plotly.min.js` | Python for the browser and the packages it needs (scikit-learn, pandas, Plotly, ...). |
| the five `.csv` files | The sample datasets for the "Try a sample" buttons. |
| `.nojekyll` | Tells GitHub Pages to publish the files as they are (a hidden file; optional). |

The whole folder is about 55 MB in 34 files (none over 15 MB). Visitors download it once (about 30 seconds); after
that their browser keeps it, so later visits start quickly.

## Put it online for free (no card)

Free, and no card needed.

### GitHub Pages

1. Create a free account at <https://github.com/signup>.
2. Click **+ > New repository**. Name it `churnsense`, choose **Public**, and click **Create repository**.
3. On the new repository page, click **uploading an existing file**.
4. Open this folder on your computer, select **all the files** (Ctrl + A), and drag them into the upload area (or
   click **choose your files** and select them all). Wait until every file is listed, then click **Commit changes**.
   - `.nojekyll` is a hidden file. If you cannot see it, skip it; the site still works.
5. Go to **Settings > Pages**. Under **Build and deployment**, set **Source** to **Deploy from a branch**, choose
   **main** and **/ (root)**, and click **Save**.
6. After about a minute, the page shows your address: `https://YOUR-USERNAME.github.io/churnsense/`.

## Add it to your Wix site

Use your address from above.

1. In the Wix Editor, go to the page.
2. Click **Add Elements (+) > Embed Code > Embed a Site** (in Wix Harmony: **+ Add > Embed > HTML Element**, then
   choose **Website Address**).
3. Paste the address and click **Apply** / **Update**.
4. Drag the box to full width and about 1400 px tall. Check the **mobile view** (phone icon) too.
5. Click **Publish**.

Tip: also add a button "Open the app in full screen" that links to the same address. It is easier to use on phones.

## Using the site

1. **Welcome screen**: visitors choose **School** or **Business**, type their name, organisation (e.g. Mayo College,
   Ajmer) and email (all three are required) and pick what they want to understand. School shows **Students, Teachers,
   Applicants** and **My own data**; Business shows **Employees, Customers, Applicants** and **My own data**. ChurnSense then takes them
   straight to a matching workspace: the right sample is loaded, the greeting and example questions fit their topic,
   and a ready-made question is in the prompt box.
   - Every choice except **My own data** opens with **test data** (a sample spreadsheet). The page says so on the
     welcome screen and in a banner on the Home page, whose **Use my institution's data** button switches to My own data.
   - **Signed in** (all three details filled in): the profile and every uploaded spreadsheet are **saved in that
     browser** (localStorage and IndexedDB), so the next visit goes straight back to them. **Sign out** (top right)
     asks first, then deletes the profile and the saved spreadsheets from the browser.
   - **Continue as guest**: no details needed, but **nothing is saved**; a banner says so, and closing the page
     clears everything. **Sign in to save** on that banner goes back to the welcome screen.
   - This is a *profile*, not a password login: everything is kept only in that visitor's browser, never on a server.
2. **Dashboard**: after the first analysis, the Home page shows five summary cards (how many people, the leave / join
   rate, the best model and its score, how many are high risk, and the top reason).
3. **Add spreadsheets**: drag `.csv` or `.xlsx` files into the box, click it to choose files, or try a sample. Each
   file appears in the list above the box with its rows and columns. **Analyse** picks the file to work on; ✕ removes it.
4. **Ask**: type a request and press **Run** (or Ctrl+Enter). For example:
   - "Which employees are most likely to leave? Target is Attrition"
   - "Why do customers churn?", "Compare the models", "Show the segments", "Is the model fair?"
   - "Explain employee 1043", "What if OverTime = No for employee 1043"
   - "Histogram of MonthlyIncome", "Churn rate by Contract", "Download the report"
   - Type **help** for the full list.
   The answer says what was done (file, target column, models trained) and shows the matching charts and tables.
5. **★ All graphs** (in the sidebar, or ask "show all graphs") shows every graph at once, grouped by model, with
   the graphs from the original notebook tagged.
6. **The sidebar pages** have every detail: Setup (target column, columns, sensitive columns, action rules),
   Explore, Train (all settings, per-model results, decision threshold), Compare, Why They Leave / Join (reasons,
   SHAP, segments, report and chart downloads), Predict & Explain (one person, what-if, ranked lists), Fairness and
   Regression.

**Share results (Word document + CSV by email):** visitors give their email address at sign-in (required, no
password). After an analysis, **Share results** (top right), **✉ Email / share results** under an answer, or
**Download the results (Word)** in the Why tab makes:

- a **Word document** (`.docx`): key numbers, top reasons, the top 20 people with their reasons, suggested actions,
  5 charts with their explanations, the model comparison and the fairness check;
- a **CSV file**: the full ranked list of the test set (each person's row, score, level and top reason).

The share window shows both attachments (click one to download it), the address and an editable message.

- **Share with attachments** (phones, and Chrome / Edge / Safari on most computers): opens the device's share menu;
  pick Outlook, Gmail or Mail and both files are attached.
- **Download files & open email** (works everywhere): saves both files to Downloads and opens the email app with the
  address, subject and message filled in; drag the two files into the email and press Send.
- Nothing is sent until the visitor presses Send. The CSV contains rows from their spreadsheet, so they should send it
  only to people allowed to see that data.
- **Optional, send from the page with EmailJS** (free, no card): the free plan sends the message **text only**, with no
  attachments; attachments need a paid EmailJS plan. To turn it on:
  1. Sign up at <https://www.emailjs.com>.
  2. **Email Services > Add New Service > Gmail** (or another), connect your account, and copy the **Service ID**.
  3. **Email Templates > Create New Template**. Set **To Email** to `{{to_email}}`, **Subject** to `{{subject}}`,
     **From Name** to `{{from_name}}`, **Reply To** to `{{reply_to}}`, and the content to `{{message}}`. Save, and
     copy the **Template ID**.
  4. **Account > General**: copy the **Public Key**.
  5. On GitHub, open `email-config.js`, click the pencil (Edit), paste the three values between the quotes, and click
     **Commit changes**. A **Send email (text only)** button then appears.

**Optional AI answers:** under **AI answers**, a visitor can paste a free Gemini API key from
<https://aistudio.google.com/apikey> (Google account needed, no card). Then any question also gets an answer from
Gemini, based on a *summary of the results* (scores, top reasons, segment and fairness summaries, and the IDs and
probabilities of the top 10 people). The spreadsheet itself is never sent. On Google's free tier, Google may use what
is sent to improve its products. Without a key, the built-in commands work as usual.

## Good to know

- **Speed:** the browser uses one CPU core. *Quick* (the default) trains all five models in about 10-40 seconds on the
  samples; *Thorough* (choose it in the Train tab, or say "use thorough") takes several minutes.
- **Explanations:** without the `shap` package in the browser, tree models (Decision Tree, Random Forest) are
  explained with exact tree path contributions, and the other models with permutation SHAP. Both add up exactly to
  each person's predicted probability.
- **Privacy:** files never leave the visitor's browser. Signed-in visitors' files are kept in that browser until they
  sign out; guests' files disappear when the page is closed.
- **Fairness:** sensitive columns (gender, age, ...) are left out of training by default and used in the fairness check.
- **Predictions show patterns, not certainties.** They must never be the only basis for a decision about a person.
- **Opening `index.html` directly from your computer does not work** (browsers block it). To try it locally, run
  `python -m http.server` in this folder and open <http://localhost:8000>.
- **Updating:** upload the changed files again (GitHub: **Add file > Upload files > Commit changes**). Wix shows the new version by itself.
