// ChurnSense mailer: lets the ChurnSense website send result emails, with the Word report and the CSV
// attached, from your Google account (Gmail). Free: about 100 emails a day on a normal Gmail account.
//
// Set up once (about 5 minutes; full steps in README.md, "Send emails directly"):
//   1. Go to https://script.google.com and click New project. Delete the code there and paste this file.
//   2. Click Save, then Run (with "setup" chosen) and allow the permissions.
//   3. Click Deploy > New deployment > Web app. Execute as: Me. Who has access: Anyone. Click Deploy.
//   4. Copy the Web app URL (it ends with /exec) into email-config.js on GitHub, as appsScriptUrl.

const MAX_EMAILS_PER_DAY = 60;      // a safety limit, below Gmail's own daily limit
const MAX_RECIPIENTS = 5;
const MAX_ATTACHMENTS = 3;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// Run this once from the editor so Google asks for permission to send email.
function setup() {
  Logger.log('ChurnSense mailer is ready. Emails left today: ' + MailApp.getRemainingDailyQuota());
}

// The website posts {to, subject, body, fromName, replyTo, attachments: [{name, type, data (base64)}]}.
function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const to = String(data.to || '').split(/[\s,;]+/).filter(String);
    if (!to.length || to.length > MAX_RECIPIENTS || !to.every(function (a) { return EMAIL_RE.test(a); })) {
      return reply({ ok: false, error: 'Please check the email address (up to ' + MAX_RECIPIENTS + ' addresses).' });
    }
    const props = PropertiesService.getScriptProperties();
    const key = 'sent-' + Utilities.formatDate(new Date(), 'GMT', 'yyyy-MM-dd');
    const sent = Number(props.getProperty(key) || 0);
    if (sent + to.length > MAX_EMAILS_PER_DAY || MailApp.getRemainingDailyQuota() < to.length) {
      return reply({ ok: false, error: 'The daily email limit is reached. Try again tomorrow, or use "Download files & open email".' });
    }
    const attachments = (data.attachments || []).slice(0, MAX_ATTACHMENTS).map(function (a) {
      return Utilities.newBlob(Utilities.base64Decode(String(a.data)), String(a.type || 'application/octet-stream'),
                               String(a.name || 'attachment').slice(0, 120));
    });
    const options = {
      to: to.join(','),
      subject: String(data.subject || 'ChurnSense results').slice(0, 200),
      body: String(data.body || '').slice(0, 20000) + '\n\n-- \nSent with ChurnSense',
      name: String(data.fromName || 'ChurnSense').slice(0, 60),
      attachments: attachments,
    };
    if (EMAIL_RE.test(String(data.replyTo || ''))) options.replyTo = String(data.replyTo);
    MailApp.sendEmail(options);
    props.setProperty(key, String(sent + to.length));
    return reply({ ok: true, sentTo: to, attachments: attachments.length });
  } catch (err) {
    return reply({ ok: false, error: String((err && err.message) || err) });
  }
}

// Opening the Web app URL in a browser shows this, to check the deployment works.
function doGet() {
  return reply({ ok: true, service: 'ChurnSense mailer' });
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
