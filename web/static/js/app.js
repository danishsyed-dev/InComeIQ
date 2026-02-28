/**
 * InComeIQ — app.js
 * Handles: form submit loading state, client-side validation, copy-to-clipboard
 */

/* ── Form Submit — Loading State & Validation ─────────────────── */
(function () {
  const form = document.getElementById('prediction-form');
  if (!form) return;

  const submitBtn = document.getElementById('submit-btn');
  const btnText   = document.getElementById('btn-text');
  const spinner   = document.getElementById('btn-spinner');

  // Validation rules  { fieldId, min, max, message }
  const rules = [
    { id: 'age',            min: 17,  max: 90,  msg: 'Age must be between 17 and 90.' },
    { id: 'hours_per_week', min: 1,   max: 99,  msg: 'Hours per week must be between 1 and 99.' },
    { id: 'capital_gain',   min: 0,   max: null, msg: 'Capital gain must be 0 or greater.' },
    { id: 'capital_loss',   min: 0,   max: null, msg: 'Capital loss must be 0 or greater.' },
  ];

  function clearErrors() {
    form.querySelectorAll('.form-group').forEach(function (g) {
      g.classList.remove('has-error');
      const inp = g.querySelector('input, select');
      if (inp) inp.classList.remove('is-invalid');
    });
  }

  function showError(fieldId, msg) {
    const input = document.getElementById(fieldId);
    if (!input) return;
    const group = input.closest('.form-group');
    if (!group) return;
    input.classList.add('is-invalid');
    group.classList.add('has-error');
    let errEl = group.querySelector('.field-error');
    if (!errEl) {
      errEl = document.createElement('span');
      errEl.className = 'field-error';
      group.appendChild(errEl);
    }
    errEl.textContent = msg;
    errEl.style.display = 'block';
  }

  function validate() {
    clearErrors();
    let valid = true;

    rules.forEach(function (rule) {
      const el = document.getElementById(rule.id);
      if (!el) return;
      const val = parseFloat(el.value);
      if (isNaN(val) || (rule.min !== null && val < rule.min) || (rule.max !== null && val > rule.max)) {
        showError(rule.id, rule.msg);
        valid = false;
      }
    });

    return valid;
  }

  form.addEventListener('submit', function (e) {
    if (!validate()) {
      e.preventDefault();
      return;
    }

    // Show loading state
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.classList.add('loading');
      if (btnText) btnText.textContent = 'Predicting…';
      if (spinner) spinner.style.display = 'inline-block';
    }
  });

  // Live clear error on input change
  form.querySelectorAll('input, select').forEach(function (el) {
    el.addEventListener('input', function () {
      el.classList.remove('is-invalid');
      const group = el.closest('.form-group');
      if (group) group.classList.remove('has-error');
    });
  });
})();

/* ── Copy Result to Clipboard ─────────────────────────────────── */
(function () {
  const copyBtn = document.getElementById('copy-result-btn');
  if (!copyBtn) return;

  const feedback = document.getElementById('copy-feedback');

  copyBtn.addEventListener('click', function () {
    const resultText = copyBtn.dataset.result || '';
    navigator.clipboard.writeText(resultText).then(function () {
      if (feedback) {
        feedback.classList.add('visible');
        setTimeout(function () { feedback.classList.remove('visible'); }, 2000);
      }
    }).catch(function () {
      // Fallback for older browsers
      const ta = document.createElement('textarea');
      ta.value = resultText;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    });
  });
})();

/* ── Confidence bar animate on load ─────────────────────────── */
(function () {
  const fill = document.getElementById('confidence-fill');
  if (!fill) return;
  const target = fill.dataset.pct || '0%';
  // Start at 0, animate to target
  fill.style.width = '0%';
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      fill.style.width = target;
    });
  });
})();
