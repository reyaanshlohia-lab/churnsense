// Optional: let ChurnSense send result emails by itself, using EmailJS (free, no card).
// Leave the three values empty and the "Email results" window opens the visitor's own email app instead.
// To turn it on (about 5 minutes), see "Email results" in README.md, then paste your three values here.
window.CHURNSENSE_EMAIL = {
  publicKey: '',    // EmailJS > Account > Public Key
  serviceId: '',    // EmailJS > Email Services > Service ID (e.g. service_abc123)
  templateId: '',   // EmailJS > Email Templates > Template ID (e.g. template_xyz789)
};
