document.addEventListener('DOMContentLoaded', () => {
  const textarea = document.getElementById('reviewText');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const resultPanel = document.getElementById('resultPanel');
  const resultTitle = document.getElementById('resultTitle');
  const scorePill = document.getElementById('scorePill');
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceValue = document.getElementById('confidenceValue');
  const inputPreview = document.getElementById('inputPreview');

  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      textarea.value = chip.dataset.example;
      textarea.focus();
    });
  });

  analyzeBtn.addEventListener('click', async () => {
    const text = textarea.value.trim();
    if (!text) {
      alert('Please enter some text to analyze.');
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      resultPanel.classList.remove('hidden');
      resultTitle.textContent = data.sentiment;
      scorePill.textContent = data.score.toFixed(3);
      scorePill.style.background = data.sentiment === 'Positive' ? 'rgba(52, 211, 153, 0.2)' : 'rgba(251, 113, 133, 0.2)';
      scorePill.style.color = data.sentiment === 'Positive' ? '#34d399' : '#fb7185';
      confidenceFill.style.width = `${data.confidence}%`;
      confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;
      inputPreview.textContent = data.text.length > 40 ? `${data.text.slice(0, 40)}...` : data.text;
    } catch (error) {
      alert(error.message);
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyze Sentiment';
    }
  });
});
