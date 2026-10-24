// Optional: let ChurnSense send result emails directly from the page.
// Leave everything empty and the share window downloads the files and opens the visitor's email app instead.
window.CHURNSENSE_EMAIL = {
  // Recommended (free, sends the Word report + CSV as attachments from your Gmail):
  // the Web app URL of churnsense-mailer.gs, ending with /exec. Steps: README.md, "Send emails directly".
  appsScriptUrl: '',

  // Alternative: EmailJS (the free plan sends the message text only, no attachments).
  publicKey: '',    // EmailJS > Account > Public Key
  serviceId: '',    // EmailJS > Email Services > Service ID
  templateId: '',   // EmailJS > Email Templates > Template ID
};
