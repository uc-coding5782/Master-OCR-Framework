/**
 * OCR Chillspace - Gen Z Cozy Front End Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const galleryBtn = document.getElementById('gallery-btn');
  const sampleBtn = document.getElementById('sample-btn');
  const filePreviewCard = document.getElementById('file-preview-card');
  const previewFilename = document.getElementById('preview-filename');
  const previewMeta = document.getElementById('preview-meta');
  const fileThumb = document.getElementById('file-thumb');
  const removeFileBtn = document.getElementById('remove-file-btn');
  const transcribeBtn = document.getElementById('transcribe-btn');
  
  const loadingState = document.getElementById('loading-state');
  const loadingText = document.getElementById('loading-text');
  const apiStatus = document.getElementById('api-status');
  
  // Carrot Box Elements
  const resultSection = document.getElementById('result-section');
  const carrotBox = document.getElementById('carrot-box');
  const carrotHeader = document.getElementById('carrot-header');
  const carrotToggleBtn = document.getElementById('carrot-toggle-btn');
  
  // Metrics
  const metricPages = document.getElementById('metric-pages');
  const metricConfidence = document.getElementById('metric-confidence');
  const metricTime = document.getElementById('metric-time');
  
  // Views
  const cozyReaderContent = document.getElementById('cozy-reader-content');
  const rawTextArea = document.getElementById('raw-text-area');
  const jsonCode = document.getElementById('json-code');
  const linesList = document.getElementById('lines-list');
  const overlayCanvas = document.getElementById('overlay-canvas');
  
  // Actions & Controls
  const ttsBtn = document.getElementById('tts-btn');
  const copyBtn = document.getElementById('copy-btn');
  const downloadTxtBtn = document.getElementById('download-txt-btn');
  const languageSelect = document.getElementById('language-select');
  const includeBoxesCheckbox = document.getElementById('include-boxes');
  
  // State
  let currentFile = null;
  let ocrResultData = null;
  let readerFontSize = 1.12; // rem
  let ttsSpeech = null;
  let ambientAudioCtx = null;
  let ambientSource = null;

  // Initialize API Health Check
  checkHealth();

  // =========================================================================
  // 1. Health Check & API Status
  // =========================================================================
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        const gpuText = data.gpu_available ? '• GPU Active' : '';
        apiStatus.classList.add('online');
        apiStatus.querySelector('.status-text').textContent = `API Ready (v${data.version}) ${gpuText}`;
      } else {
        apiStatus.querySelector('.status-text').textContent = 'API Unavailable';
      }
    } catch (e) {
      apiStatus.querySelector('.status-text').textContent = 'API Offline';
    }
  }

  // =========================================================================
  // 2. Upload Zone & File Handling (Gallery / Sample / Drag & Drop)
  // =========================================================================
  galleryBtn.addEventListener('click', () => fileInput.click());

  if (sampleBtn) {
    sampleBtn.addEventListener('click', async () => {
      try {
        const res = await fetch('/static/sample.png');
        const blob = await res.blob();
        const sampleFile = new File([blob], 'sample_test.png', { type: 'image/png' });
        handleFileSelected(sampleFile);
      } catch (err) {
        alert('Failed to load sample image: ' + err.message);
      }
    });
  }

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag & Drop Events
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-over');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt.files && dt.files[0]) {
      handleFileSelected(dt.files[0]);
    }
  });

  function handleFileSelected(file) {
    currentFile = file;
    previewFilename.textContent = file.name;
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    const ext = file.name.split('.').pop().toUpperCase();
    previewMeta.textContent = `${sizeMB} MB • ${ext} File`;

    if (file.type.startsWith('image/')) {
      fileThumb.textContent = '🖼️';
    } else if (file.type.includes('pdf')) {
      fileThumb.textContent = '📚';
    } else {
      fileThumb.textContent = '📄';
    }

    dropzone.classList.add('hidden');
    filePreviewCard.classList.remove('hidden');
    resultSection.classList.add('hidden');
  }

  removeFileBtn.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    filePreviewCard.classList.add('hidden');
    dropzone.classList.remove('hidden');
    resultSection.classList.add('hidden');
  });

  // =========================================================================
  // 3. Perform OCR Request
  // =========================================================================
  transcribeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    filePreviewCard.classList.add('hidden');
    loadingState.classList.remove('hidden');
    resultSection.classList.add('hidden');

    const isPdf = currentFile.name.toLowerCase().endsWith('.pdf') || 
                  currentFile.name.toLowerCase().endsWith('.tif') || 
                  currentFile.name.toLowerCase().endsWith('.tiff');

    const endpoint = isPdf ? '/ocr/pdf' : '/ocr/image';
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('language', languageSelect.value);
    formData.append('include_boxes', includeBoxesCheckbox.checked ? 'true' : 'false');

    const startTime = performance.now();

    try {
      loadingText.textContent = 'Analyzing pages & extracting text... ☕';
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'OCR Processing failed');
      }

      const result = await res.json();
      const duration = ((performance.now() - startTime) / 1000).toFixed(2);
      
      ocrResultData = result;
      displayCarrotBoxResults(result, duration);

    } catch (err) {
      alert(`OCR Error: ${err.message}`);
      filePreviewCard.classList.remove('hidden');
    } finally {
      loadingState.classList.add('hidden');
    }
  });

  // =========================================================================
  // 4. Render Carrot Box & Results
  // =========================================================================
  function displayCarrotBoxResults(result, duration) {
    // 1. Calculate Aggregate Metrics
    const totalPages = result.page_count || (result.pages ? result.pages.length : 1);
    let totalConf = 0;
    let lineCount = 0;
    let fullTextArr = [];
    let allLines = [];

    if (result.pages && result.pages.length > 0) {
      result.pages.forEach(page => {
        if (page.lines) {
          page.lines.forEach(l => {
            fullTextArr.push(l.text);
            totalConf += (l.confidence || 0);
            lineCount++;
            allLines.push(l);
          });
        }
      });
    }

    const avgConf = lineCount > 0 ? ((totalConf / lineCount) * 100).toFixed(1) : '95.0';

    // Update Header Badges
    metricPages.textContent = `${totalPages} Page${totalPages > 1 ? 's' : ''}`;
    metricConfidence.textContent = `Conf: ${avgConf}%`;
    metricTime.textContent = `${duration}s`;

    // 2. Render Cozy Reader Content
    const fullText = fullTextArr.join('\n\n');
    cozyReaderContent.textContent = fullText || 'No text recognized in document.';
    rawTextArea.value = fullText;
    jsonCode.textContent = JSON.stringify(result, null, 2);

    // Update Output File Banner Stats & Name
    const statWords = document.getElementById('stat-words');
    const statLines = document.getElementById('stat-lines');
    const statChars = document.getElementById('stat-chars');
    const outputFilename = document.getElementById('output-filename');
    const exportFormatSelect = document.getElementById('export-format-select');

    const words = fullText.trim().split(/\s+/).filter(Boolean).length;
    const chars = fullText.length;

    if (statWords) statWords.textContent = `${words} Words`;
    if (statLines) statLines.textContent = `${lineCount} Lines`;
    if (statChars) statChars.textContent = `${chars} Chars`;

    const baseName = currentFile ? (currentFile.name.substring(0, currentFile.name.lastIndexOf('.')) || 'transcribed_doc') : 'transcribed_doc';
    const ext = exportFormatSelect ? exportFormatSelect.value : 'txt';
    if (outputFilename) outputFilename.textContent = `${baseName}_transcript.${ext}`;

    // 3. Render Line Breakdown List
    linesList.innerHTML = '';
    allLines.forEach((line, idx) => {
      const confPct = Math.round((line.confidence || 0) * 100);
      const bboxStr = line.bounding_box ? 
        `[${Math.round(line.bounding_box.x_min)}, ${Math.round(line.bounding_box.y_min)}, ${Math.round(line.bounding_box.x_max)}, ${Math.round(line.bounding_box.y_max)}]` : 
        'No bbox';

      const card = document.createElement('div');
      card.className = 'line-item-card';
      card.innerHTML = `
        <span class="line-text-str">${idx + 1}. ${escapeHtml(line.text)}</span>
        <div class="line-meta-row">
          <span><span class="conf-bar" style="width: ${confPct / 2}px;"></span>${confPct}%</span>
          <span>${line.engine || 'OCR'} • ${bboxStr}</span>
        </div>
      `;
      linesList.appendChild(card);
    });

    // 4. Render Bounding Box Overlay Canvas (for images)
    if (currentFile && (!currentFile.type || currentFile.type.startsWith('image/') || currentFile.name.match(/\.(png|jpe?g|webp|bmp|tiff?)$/i))) {
      renderCanvasOverlay(allLines);
    }

    // Expand Carrot Box & Show Result Section
    carrotBox.classList.remove('collapsed');
    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth' });
  }

  // =========================================================================
  // 5. Carrot Box Accordion Toggle
  // =========================================================================
  carrotHeader.addEventListener('click', toggleCarrotBox);
  carrotToggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleCarrotBox();
  });

  function toggleCarrotBox() {
    carrotBox.classList.toggle('collapsed');
  }

  // =========================================================================
  // 6. View Tabs Switcher
  // =========================================================================
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(`pane-${btn.dataset.tab}`);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // =========================================================================
  // 7. Cozy Reader Controls (Font, Size, Theme)
  // =========================================================================
  const bookPageSheet = document.getElementById('book-page-sheet');

  document.querySelectorAll('.font-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.font-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (bookPageSheet) {
        bookPageSheet.classList.remove('font-serif', 'font-sans');
        bookPageSheet.classList.add(`font-${btn.dataset.font}`);
      }
    });
  });

  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (bookPageSheet) {
        bookPageSheet.classList.remove('theme-dark', 'theme-sepia', 'theme-paper');
        bookPageSheet.classList.add(`theme-${btn.dataset.theme}`);
      }
    });
  });

  document.getElementById('size-inc').addEventListener('click', () => {
    readerFontSize = Math.min(readerFontSize + 0.1, 1.8);
    cozyReaderContent.style.fontSize = `${readerFontSize}rem`;
  });

  document.getElementById('size-dec').addEventListener('click', () => {
    readerFontSize = Math.max(readerFontSize - 0.1, 0.85);
    cozyReaderContent.style.fontSize = `${readerFontSize}rem`;
  });

  // =========================================================================
  // 8. Bounding Box Overlay Canvas Drawing
  // =========================================================================
  function renderCanvasOverlay(lines) {
    if (!currentFile) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        overlayCanvas.width = img.width;
        overlayCanvas.height = img.height;
        const ctx = overlayCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        // Draw Bounding Boxes
        lines.forEach(l => {
          if (l.bounding_box) {
            const { x_min, y_min, x_max, y_max } = l.bounding_box;
            const w = x_max - x_min;
            const h = y_max - y_min;

            // Box stroke
            ctx.strokeStyle = '#f97316';
            ctx.lineWidth = Math.max(2, Math.round(img.width / 400));
            ctx.strokeRect(x_min, y_min, w, h);

            // Semi-transparent fill
            ctx.fillStyle = 'rgba(249, 115, 22, 0.15)';
            ctx.fillRect(x_min, y_min, w, h);
          }
        });
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(currentFile);
  }

  // =========================================================================
  // 9. Actions: Copy, Multi-Format Download, Text-to-Speech (TTS)
  // =========================================================================
  const downloadFileBtn = document.getElementById('download-file-btn');
  const exportFormatSelect = document.getElementById('export-format-select');
  const outputFilename = document.getElementById('output-filename');

  if (exportFormatSelect && outputFilename) {
    exportFormatSelect.addEventListener('change', () => {
      const baseName = currentFile ? (currentFile.name.substring(0, currentFile.name.lastIndexOf('.')) || 'transcribed_doc') : 'transcribed_doc';
      outputFilename.textContent = `${baseName}_transcript.${exportFormatSelect.value}`;
    });
  }

  if (downloadFileBtn) {
    downloadFileBtn.addEventListener('click', triggerMultiFormatDownload);
  }

  if (downloadTxtBtn) {
    downloadTxtBtn.addEventListener('click', triggerMultiFormatDownload);
  }

  function triggerMultiFormatDownload() {
    if (!ocrResultData) return;
    const format = exportFormatSelect ? exportFormatSelect.value : 'txt';
    let baseName = 'transcribed_doc';
    if (currentFile && currentFile.name) {
      const dotIdx = currentFile.name.lastIndexOf('.');
      baseName = dotIdx > 0 ? currentFile.name.substring(0, dotIdx) : currentFile.name;
    }
    // Clean baseName of any illegal filename characters
    baseName = baseName.replace(/[\\/:*?"<>|]/g, '_');
    const fileName = `${baseName}_transcript.${format}`;

    let content = '';
    let mimeType = 'text/plain;charset=utf-8';

    if (format === 'txt') {
      content = rawTextArea.value || cozyReaderContent.textContent;
    } else if (format === 'md') {
      content = `# Transcribed Document: ${baseName}\n\n` +
                `* **Language**: ${languageSelect.value}\n` +
                `* **Pages**: ${ocrResultData.page_count || 1}\n\n` +
                `---\n\n` + (rawTextArea.value || cozyReaderContent.textContent);
      mimeType = 'text/markdown;charset=utf-8';
    } else if (format === 'csv') {
      let csvRows = ['"Line","Text","Confidence","Engine","BoundingBox"'];
      let idx = 1;
      if (ocrResultData.pages) {
        ocrResultData.pages.forEach(p => {
          if (p.lines) {
            p.lines.forEach(l => {
              const escapedText = (l.text || '').replace(/"/g, '""');
              const bboxStr = l.bounding_box ? `[${l.bounding_box.x_min},${l.bounding_box.y_min},${l.bounding_box.x_max},${l.bounding_box.y_max}]` : '';
              csvRows.push(`"${idx}","${escapedText}","${l.confidence || 1.0}","${l.engine || 'OCR'}","${bboxStr}"`);
              idx++;
            });
          }
        });
      }
      content = csvRows.join('\n');
      mimeType = 'text/csv;charset=utf-8';
    } else if (format === 'json') {
      content = JSON.stringify(ocrResultData, null, 2);
      mimeType = 'application/json;charset=utf-8';
    }

    // Direct client-side download: guarantees clean, exact filename without server UUIDs or popup blockers
    downloadBlobFile(content, fileName, mimeType);
  }

  function downloadBlobFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 200);
  }

  copyBtn.addEventListener('click', () => {
    const text = cozyReaderContent.textContent;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const originalText = copyBtn.querySelector('.btn-text').textContent;
      copyBtn.querySelector('.btn-text').textContent = 'Copied! ✨';
      setTimeout(() => {
        copyBtn.querySelector('.btn-text').textContent = originalText;
      }, 2000);
    });
  });

  // Text to Speech
  ttsBtn.addEventListener('click', () => {
    if (!('speechSynthesis' in window)) {
      alert('Speech synthesis not supported on this browser.');
      return;
    }

    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      ttsBtn.classList.remove('speaking');
      ttsBtn.querySelector('.btn-text').textContent = 'Listen';
      return;
    }

    const text = cozyReaderContent.textContent;
    if (!text) return;

    ttsSpeech = new SpeechSynthesisUtterance(text);
    ttsSpeech.rate = 0.95;
    ttsSpeech.pitch = 1.0;

    ttsSpeech.onstart = () => {
      ttsBtn.classList.add('speaking');
      ttsBtn.querySelector('.btn-text').textContent = 'Pause';
    };

    ttsSpeech.onend = ttsSpeech.onerror = () => {
      ttsBtn.classList.remove('speaking');
      ttsBtn.querySelector('.btn-text').textContent = 'Listen';
    };

    window.speechSynthesis.speak(ttsSpeech);
  });

  // =========================================================================
  // 10. YouTube Ambient Radio Stream Player Integration
  // =========================================================================
  const ytPlayer = document.getElementById('yt-player');
  document.querySelectorAll('.sound-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sound-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const ytId = btn.dataset.yt;
      if (!ytId || ytId === 'none') {
        if (ytPlayer) ytPlayer.src = '';
      } else {
        if (ytPlayer) ytPlayer.src = `https://www.youtube.com/embed/${ytId}?autoplay=1&enablejsapi=1&loop=1&playlist=${ytId}`;
      }
    });
  });

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
});
