document.getElementById("prompt").addEventListener("input", function () {
    updateSplitButtonStatus();
    updatePromptCharCount();
    updatePromptTokenCount();
  });

  document.getElementById('split_length').addEventListener('input', () => { updateSplitButtonStatus(); updatePromptTokenCount(); });

  function copyToClipboard(element) {
    const textArea = document.createElement("textarea");
    textArea.value = element.getAttribute("data-content");
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);
    element.classList.add("clicked");
  }

  function copyInstructions() {
    const instructionsButton = document.getElementById("copy-instructions-btn");
    const instructions = document.getElementById("instructions").textContent;
    const textArea = document.createElement("textarea");
    textArea.value = instructions;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);
    instructionsButton.classList.add("clicked");
  }

  function toggleCustomLength(select) {
    const customLengthInput = document.getElementById("split_length");
    if (select.value === "custom") {
      customLengthInput.style.display = "inline";
    } else {
      customLengthInput.value = select.value;
      customLengthInput.style.display = "none";
    }
  }

  // Mode handling: chars vs tokens
  const modeRadios = document.getElementsByName('mode');
  Array.from(modeRadios).forEach(r => {
    r.addEventListener('change', () => {
      const encodingSelect = document.getElementById('encoding');
      if (r.value === 'tokens' && r.checked) {
        encodingSelect.removeAttribute('disabled');
      } else if (r.value === 'chars' && r.checked) {
        encodingSelect.setAttribute('disabled', 'disabled');
      }
      updateSplitButtonStatus();
      updatePromptTokenCount();
    });
  });
  const encodingSelect = document.getElementById('encoding');
  if (encodingSelect) {
    encodingSelect.addEventListener('change', updatePromptTokenCount);
  }

  function updateSplitButtonStatus() {
    const promptField = document.getElementById('prompt');
    const splitLength = document.getElementById('split_length');
    const splitBtn = document.getElementById('split-btn');
    const mode = document.querySelector('input[name="mode"]:checked')?.value || 'chars';
    const promptLength = promptField.value.trim().length;
    const splitLengthValue = parseInt(splitLength.value);

    if (promptLength === 0) {
        splitBtn.setAttribute('disabled', 'disabled');
        splitBtn.classList.add('disabled');
        splitBtn.textContent = 'Enter a prompt';
    } else if (isNaN(splitLengthValue) || splitLengthValue === 0) {
        splitBtn.setAttribute('disabled', 'disabled');
        splitBtn.classList.add('disabled');
        splitBtn.textContent = 'Enter the length for calculating';
    } else if (mode === 'chars' && promptLength < splitLengthValue) {
        splitBtn.setAttribute('disabled', 'disabled');
        splitBtn.classList.add('disabled');
        splitBtn.textContent = 'Prompt is shorter than split length';
    } else {
        splitBtn.removeAttribute('disabled');
        splitBtn.classList.remove('disabled');
        if (mode === 'chars') {
          splitBtn.textContent = `Split into ${Math.ceil(promptLength / splitLengthValue)} parts`;
        } else {
          const tokenCount = parseInt(document.getElementById('prompt-token-count').textContent) || 0;
          if (tokenCount === 0) {
            splitBtn.textContent = 'Calculating tokens…';
            splitBtn.setAttribute('disabled', 'disabled');
            splitBtn.classList.add('disabled');
          } else {
            splitBtn.textContent = `Split into ${Math.ceil(tokenCount / splitLengthValue)} parts`;
          }
        }
    }
  }

  function updatePromptCharCount() {
    const promptField = document.getElementById("prompt");
    const charCount = document.getElementById("prompt-char-count");
    const promptLength = promptField.value.trim().length;
    charCount.textContent = promptLength;
  }

  async function updatePromptTokenCount() {
    const tokenEl = document.getElementById('prompt-token-count');
    if (!tokenEl) return;
    const mode = document.querySelector('input[name="mode"]:checked')?.value || 'chars';
    // Always compute tokens so user can see count even in Characters mode
    const promptField = document.getElementById('prompt');
    const encoding = document.getElementById('encoding')?.value || 'cl100k_base';
    const text = promptField.value || '';
    if (!text.trim()) {
      tokenEl.textContent = '0';
      return;
    }
    try {
      const res = await fetch('/api/count_tokens', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, encoding })
      });
      const data = await res.json();
      tokenEl.textContent = data.count ?? '0';
    } catch (e) {
      tokenEl.textContent = '0';
    } finally {
      updateSplitButtonStatus();
    }
  }

  updateSplitButtonStatus();
  updatePromptTokenCount();