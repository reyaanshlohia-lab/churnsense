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
| `email-config.js` | Settings for sending result emails directly (see **Send emails directly** below). |
| `churnsense-mailer.gs` | The free Google Apps Script that sends those emails with attachments (pasted into script.google.com, not run by the page). |
| `help.js` | The Help chat: explains every chart (each line, colour and star), the words used, and how to use the app. |
| `worker.js` | Runs the Python analysis engine in the background, so the page never freezes. |
| `engine.py` | The analysis code (the same as `people_prediction_app.ipynb`, adapted for the browser). |
| `pyodide.*`, `python_stdlib.zip`, `pyodide-lock.json`, the `.whl` files, `plotly.min.js` | Python for the browser and the packages it needs (scikit-learn, pandas, Plotly, ...). |
| the five `.csv` files | The sample datasets for the "Try a sample" buttons: 10,000+ rows and 50+ columns each, all marked `_SYNTHETIC` (employees and customers keep every row of the real IBM datasets, plus made-up rows and columns). In the students file, high achievers also leave, for schools that fit their career goal better. |
| `.nojekyll` | Tells GitHub Pages to publish the files as they are (a hidden file; optional). |

The whole folder is about 55 MB in 36 files (none over 15 MB). Visitors download it once (about 30 seconds); after
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
- **Send emails directly** (recommended, free): see the next section. The window's main button then says **Send
  email** and sends the message with both files attached, straight from the page.
- **Alternative: EmailJS** (free plan = message text only, no attachments): fill in `publicKey`, `serviceId` and
  `templateId` in `email-config.js` (template fields `{{to_email}}`, `{{subject}}`, `{{from_name}}`, `{{reply_to}}`,
  `{{message}}`). A **Send email (text only)** button then appears.

### Send emails directly (with the Word report and CSV attached)

A website without a server cannot send email by itself, so ChurnSense uses a small free **Google Apps Script**
(`churnsense-mailer.gs`) that sends through Gmail. Tip: create a separate Gmail account for the project (for example
`churnsense.team@gmail.com`), because the emails are sent from the account that sets this up.

1. Sign in to that Google account and open <https://script.google.com>. Click **New project**.
2. Delete the code in the editor, open `churnsense-mailer.gs` from this folder, copy all of it, and paste it in.
   Click the project name (top left) and call it `ChurnSense mailer`. Click **Save** (the disk icon).
3. In the toolbar, choose the function **setup** and click **Run**. Click **Review permissions**, pick your account,
   then **Advanced > Go to ChurnSense mailer (unsafe)** and **Allow**. (Google shows this warning for every personal
   script; it is your own code.)
4. Click **Deploy > New deployment**. Click the gear next to "Select type" and choose **Web app**. Set **Execute as:
   Me** and **Who has access: Anyone**. Click **Deploy**, then copy the **Web app URL** (it ends with `/exec`).
5. Check it: open that URL in a browser tab. It should show `{"ok":true,"service":"ChurnSense mailer"}`.
6. On GitHub, open `email-config.js`, click the pencil (Edit), paste the URL between the quotes of
   `appsScriptUrl: ''`, and click **Commit changes**. After about a minute, the share window's button says
   **Send email**.

Limits and safety: the script sends at most 60 emails a day and 5 addresses per email (change
`MAX_EMAILS_PER_DAY` at its top if needed; Gmail itself allows about 100 a day). Anyone who finds the Web app URL could
use it within those limits, which is another reason to use a separate project Gmail account. If you change the
script later, use **Deploy > Manage deployments > Edit (pencil) > Version: New version > Deploy** so the URL stays
the same.

**Help chat:** the **? Help** button (bottom right) opens a chat that explains what is on the page, for example
"What does the black line in the first chart mean?" (the error bars of the top-reasons chart), "What is ROC-AUC?",
or a pasted sentence such as "recall 68%, precision 41%". Every chart also has a **? Explain** button that describes
each of its parts. It works in the browser with no AI service (it reads the charts on the page and a built-in
knowledge base), so it is free, instant and private. If the visitor has entered a Gemini key, questions it cannot
match are passed to Gemini.

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
  Then raise the version number in `index.html` (the four `?v=14` in the lines near the top, e.g. to `?v=15`) and upload it
  too: browsers then download the new files instead of reusing old saved copies, which can otherwise mix old and new
  files and stop the page from starting. Wait until the **Actions** tab shows a green tick before testing.
