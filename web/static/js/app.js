/**
 * InComeIQ — app.js
 * Handles: form submit loading state, client-side validation, copy-to-clipboard,
 * quick demo presets loading, and history drawer overlay sync.
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
      var inp = g.querySelector('input, select');
      if (inp) inp.classList.remove('is-invalid');
    });
  }

  function showError(fieldId, msg) {
    var input = document.getElementById(fieldId);
    if (!input) return;
    var group = input.closest('.form-group');
    if (!group) return;
    input.classList.add('is-invalid');
    group.classList.add('has-error');
    var errEl = group.querySelector('.field-error');
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
    var valid = true;

    rules.forEach(function (rule) {
      var el = document.getElementById(rule.id);
      if (!el) return;
      var val = parseFloat(el.value);
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
      if (btnText) btnText.textContent = 'Predicting\u2026';
      if (spinner) spinner.style.display = 'inline-block';
    }
  });

  // Live clear error on input change
  form.querySelectorAll('input, select').forEach(function (el) {
    el.addEventListener('input', function () {
      el.classList.remove('is-invalid');
      var group = el.closest('.form-group');
      if (group) group.classList.remove('has-error');
    });
  });
})();

/* ── Copy Result to Clipboard ─────────────────────────────────── */
(function () {
  var copyBtn = document.getElementById('copy-result-btn');
  if (!copyBtn) return;

  var feedback = document.getElementById('copy-feedback');

  copyBtn.addEventListener('click', function () {
    var resultText = copyBtn.dataset.result || '';
    navigator.clipboard.writeText(resultText).then(function () {
      if (feedback) {
        feedback.classList.add('visible');
        setTimeout(function () { feedback.classList.remove('visible'); }, 2000);
      }
    }).catch(function () {
      // Fallback for older browsers
      var ta = document.createElement('textarea');
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

/* ── Confidence bar animate on load (GPU-accelerated scaleX) ── */
(function () {
  var fill = document.getElementById('confidence-fill');
  if (!fill) return;
  var target = fill.dataset.pct || '0%';
  var pctValue = parseFloat(target) / 100;
  // Start at 0, animate to target via transform
  fill.style.transform = 'scaleX(0)';
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      fill.style.transform = 'scaleX(' + pctValue + ')';
    });
  });
})();

/* ── Load Demo Presets ─────────────────────────────────────────── */
(function () {
  const presets = {
    software_engineer: {
      age: 32,
      sex: "1",
      race: "4",
      marital_status: "4",
      relationship: "1",
      native_country: "38",
      education_num: "13",
      workclass: "3",
      occupation: "9",
      hours_per_week: 40,
      capital_gain: 0,
      capital_loss: 0,
      model: "xgboost"
    },
    executive: {
      age: 48,
      sex: "1",
      race: "4",
      marital_status: "2",
      relationship: "0",
      native_country: "38",
      education_num: "14",
      workclass: "3",
      occupation: "3",
      hours_per_week: 50,
      capital_gain: 15000,
      capital_loss: 0,
      model: "xgboost"
    },
    retail_worker: {
      age: 21,
      sex: "0",
      race: "2",
      marital_status: "4",
      relationship: "3",
      native_country: "38",
      education_num: "10",
      workclass: "3",
      occupation: "11",
      hours_per_week: 25,
      capital_gain: 0,
      capital_loss: 0,
      model: "random_forest"
    }
  };

  document.querySelectorAll('.preset-card').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const presetName = btn.dataset.preset;
      const data = presets[presetName];
      if (!data) return;

      Object.keys(data).forEach(function (key) {
        const input = document.getElementById(key);
        if (input) {
          input.value = data[key];
          // Clear error states on preset load
          input.classList.remove('is-invalid');
          const group = input.closest('.form-group');
          if (group) group.classList.remove('has-error');
        } else {
          // Check if it is a radio button (ML model)
          const radio = document.querySelector(`input[name="${key}"][value="${data[key]}"]`);
          if (radio) {
            radio.checked = true;
          }
        }
      });
    });
  });
})();

/* ── Prediction History Drawer UI Sync ────────────────────────── */
(function () {
  const drawer = document.getElementById('history-drawer');
  const openBtn = document.getElementById('open-history-btn');
  const closeBtn = document.getElementById('close-history-btn');
  const overlay = document.getElementById('history-drawer-overlay');
  const body = document.getElementById('history-drawer-body');
  const clearBtn = document.getElementById('clear-history-btn');

  if (!drawer || !openBtn) return;

  const workclass_map = {
    0: "Federal Govt", 1: "Local Govt", 2: "Never Worked",
    3: "Private", 4: "Self-employed (Inc)", 5: "Self-employed (Not Inc)",
    6: "State Govt", 7: "Without Pay"
  };
  const education_map = {
    1: "Preschool", 2: "1st-4th Grade", 3: "5th-6th Grade", 4: "7th-8th Grade",
    5: "9th Grade", 6: "10th Grade", 7: "11th Grade", 8: "12th Grade",
    9: "High School Grad", 10: "Some College", 11: "Assoc (Voc)",
    12: "Assoc (Acad)", 13: "Bachelors", 14: "Masters",
    15: "Prof School", 16: "Doctorate"
  };
  const marital_map = {
    0: "Divorced", 1: "Married (Armed Forces)", 2: "Married (Civ)",
    3: "Married (Absent)", 4: "Never Married", 5: "Separated", 6: "Widowed"
  };
  const occupation_map = {
    0: "Admin / Clerical", 1: "Armed Forces", 2: "Craft / Repair",
    3: "Exec / Managerial", 4: "Farming / Fishing", 5: "Handlers / Cleaners",
    6: "Machine Op", 7: "Other Service", 8: "Private Household",
    9: "Prof Specialty", 10: "Protective Services", 11: "Sales",
    12: "Tech Support", 13: "Transport / Moving"
  };

  function toggleDrawer(open) {
    if (open) {
      drawer.classList.add('open');
      drawer.setAttribute('aria-hidden', 'false');
      fetchHistory();
    } else {
      drawer.classList.remove('open');
      drawer.setAttribute('aria-hidden', 'true');
    }
  }

  openBtn.addEventListener('click', function () { toggleDrawer(true); });
  if (closeBtn) closeBtn.addEventListener('click', function () { toggleDrawer(false); });
  if (overlay) overlay.addEventListener('click', function () { toggleDrawer(false); });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && drawer.classList.contains('open')) {
      toggleDrawer(false);
    }
  });

  function fetchHistory() {
    if (!body) return;
    body.innerHTML = '<p class="history-empty">Loading prediction history...</p>';

    fetch('/api/history?limit=15')
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.status === 'success' && data.count > 0) {
          body.innerHTML = '';
          data.data.forEach(function (item) {
            const el = document.createElement('div');
            el.className = 'history-item';

            const isPositive = item.outputs.prediction === 1;
            const resultClass = isPositive ? 'positive' : 'negative';
            const resultLabel = isPositive ? '>50K / year' : '<=50K / year';
            const confidence = item.outputs.confidence ? (item.outputs.confidence * 100).toFixed(1) + '%' : 'N/A';
            
            // Format nice relative timestamp or localized date
            const date = new Date(item.created_at).toLocaleString();

            const workclassVal = workclass_map[item.inputs.workclass] || item.inputs.workclass;
            const educationVal = education_map[item.inputs.education_num] || item.inputs.education_num;
            const maritalVal = marital_map[item.inputs.marital_status] || item.inputs.marital_status;
            const occupationVal = occupation_map[item.inputs.occupation] || item.inputs.occupation;

            el.innerHTML = `
              <div class="history-item-header">
                <span>${date}</span>
                <span>Conf: ${confidence}</span>
              </div>
              <div class="history-item-result ${resultClass}">
                Outcome: ${resultLabel}
              </div>
              <div class="history-item-details">
                <span>Age: <strong>${item.inputs.age}</strong></span>
                <span>Hours/Wk: <strong>${item.inputs.hours_per_week}h</strong></span>
                <span>Education: <strong>${educationVal}</strong></span>
                <span>Work: <strong>${workclassVal}</strong></span>
                <span>Occupation: <strong>${occupationVal}</strong></span>
                <span>Marital: <strong>${maritalVal}</strong></span>
                <span>Gains: <strong>$${item.inputs.capital_gain.toLocaleString()}</strong></span>
                <span>Losses: <strong>$${item.inputs.capital_loss.toLocaleString()}</strong></span>
              </div>
            `;
            body.appendChild(el);
          });
        } else {
          body.innerHTML = '<p class="history-empty">No prediction history found.</p>';
        }
      })
      .catch(function () {
        body.innerHTML = '<p class="history-empty" style="color:var(--error)">Failed to load history drawer.</p>';
      });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      if (!confirm('Are you sure you want to clear all prediction history? This action is permanent.')) return;

      fetch('/api/history', { method: 'DELETE' })
        .then(function (res) { return res.json(); })
        .then(function (data) {
          if (data.status === 'success') {
            fetchHistory();
          } else {
            alert('Error clearing database history: ' + (data.error || 'Unknown error'));
          }
        })
        .catch(function (err) {
          alert('Request failed: ' + err);
        });
    });
  }
})();
