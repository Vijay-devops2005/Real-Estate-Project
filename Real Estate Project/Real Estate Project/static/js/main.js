// ─── Particles ────────────────────────────────────────────────────────────────
(function createParticles() {
  const colors = ['#e94560','#06d6a0','#ffd166','#7c3aed','#0f3460'];
  for (let i = 0; i < 20; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 4 + 1;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random()*100}vw;
      background:${colors[Math.floor(Math.random()*colors.length)]};
      opacity:${Math.random()*0.5+0.1};
      animation-duration:${Math.random()*15+10}s;
      animation-delay:${Math.random()*10}s;
    `;
    document.body.appendChild(p);
  }
})();

// ─── Slider Sync ─────────────────────────────────────────────────────────────
document.querySelectorAll('input[type="range"]').forEach(slider => {
  const display = document.getElementById(slider.id + '_val');
  if (display) {
    slider.addEventListener('input', () => {
      display.textContent = parseFloat(slider.value).toFixed(
        slider.step && parseFloat(slider.step) < 1 ? 1 : 0
      );
    });
  }
});

// ─── Toast ─────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toast-msg');
  if (!toast) return;
  toastMsg.textContent = msg;
  toast.className = `toast ${type} show`;
  setTimeout(() => toast.className = 'toast', 3500);
}

// ─── Form Data Collector ────────────────────────────────────────────────────
function collectFormData() {
  const get = id => parseFloat(document.getElementById(id)?.value) || 0;
  return {
    MedInc:     get('MedInc'),
    HouseAge:   get('HouseAge'),
    AveRooms:   get('AveRooms'),
    AveBedrms:  get('AveBedrms'),
    Population: get('Population'),
    AveOccup:   get('AveOccup'),
    Latitude:   get('Latitude'),
    Longitude:  get('Longitude'),
  };
}

// ─── Animated Counter ──────────────────────────────────────────────────────
function animateCounter(el, target, duration = 1200, prefix = '$', suffix = '') {
  const start = performance.now();
  const startVal = 0;
  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(startVal + (target - startVal) * eased);
    el.textContent = prefix + current.toLocaleString() + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ─── Predict ──────────────────────────────────────────────────────────────
async function predict() {
  const btn = document.getElementById('predict-btn');
  const placeholder = document.getElementById('result-placeholder');
  const display = document.getElementById('result-display');

  btn.classList.add('loading');
  btn.disabled = true;

  const data = collectFormData();

  // Validation
  if (data.MedInc <= 0 || data.AveRooms <= 0) {
    showToast('Please fill in all fields correctly.', 'error');
    btn.classList.remove('loading');
    btn.disabled = false;
    return;
  }

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const json = await res.json();

    if (json.error) throw new Error(json.error);

    // Hide placeholder, show result
    placeholder.style.display = 'none';
    display.classList.add('visible');

    // Animate price
    const priceEl = document.getElementById('pred-price');
    animateCounter(priceEl, json.predicted_price, 1400);

    // Fill metrics
    document.getElementById('pred-model').textContent = json.model_used;
    document.getElementById('pred-r2').textContent = (json.r2_score * 100).toFixed(1) + '%';

    // Confidence bar (map R² to width)
    const confFill = document.getElementById('conf-fill');
    const confPct = document.getElementById('conf-pct');
    const confWidth = Math.min(100, json.r2_score * 100);
    setTimeout(() => {
      confFill.style.width = confWidth + '%';
      confPct.textContent = confWidth.toFixed(1) + '%';
    }, 200);

    showToast('Prediction complete! 🏡', 'success');

  } catch (err) {
    showToast('Error: ' + err.message, 'error');
    console.error(err);
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ─── Attach predict button ──────────────────────────────────────────────────
document.getElementById('predict-btn')?.addEventListener('click', predict);

// ─── Lightbox ──────────────────────────────────────────────────────────────
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');

document.querySelectorAll('.chart-card img').forEach(img => {
  img.style.cursor = 'zoom-in';
  img.addEventListener('click', () => {
    lightboxImg.src = img.src;
    lightbox?.classList.add('open');
  });
});

document.getElementById('lightbox-close')?.addEventListener('click', () => {
  lightbox?.classList.remove('open');
});
lightbox?.addEventListener('click', e => {
  if (e.target === lightbox) lightbox.classList.remove('open');
});

// ─── Keyboard esc for lightbox ─────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') lightbox?.classList.remove('open');
});

// ─── Animate stagger for cards ──────────────────────────────────────────────
document.querySelectorAll('.card, .feature-card, .metric-card, .chart-card').forEach((el, i) => {
  el.style.animationDelay = `${i * 0.07}s`;
});
