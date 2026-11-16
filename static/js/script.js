/**
 * Real-Time Fake Job Detection System
 * Frontend JavaScript Controller
 */

document.addEventListener("DOMContentLoaded", function () {
  // -------------------------------------------------------------
  // Recruitment Poster Upload & OCR Extraction
  // -------------------------------------------------------------
  const dropzone = document.getElementById("poster-dropzone");
  const fileInput = document.getElementById("poster-file-input");
  const previewArea = document.getElementById("poster-preview-area");
  const previewImg = document.getElementById("poster-preview-img");
  const uploadPrompt = document.getElementById("upload-prompt");
  const ocrLoading = document.getElementById("ocr-loading-spinner");
  const ocrNotice = document.getElementById("ocr-notice-alert");
  const ocrNoticeText = document.getElementById("ocr-notice-text");
  const removeBtn = document.getElementById("remove-poster-btn");

  if (dropzone && fileInput) {
    // Click on dropzone opens file selector
    dropzone.addEventListener("click", function (e) {
      if (e.target === fileInput || e.target.closest("#poster-preview-area")) return;
      fileInput.click();
    });

    dropzone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    // Drag-over styling
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    // Handle dropped files
    dropzone.addEventListener("drop", function (e) {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFileSelection(files[0]);
      }
    });

    // Handle standard file selection
    fileInput.addEventListener("change", function () {
      if (fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });

    // Remove image handler
    if (removeBtn) {
      removeBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        resetUpload();
      });
    }
  }

  function handleFileSelection(file) {
    const extension = file.name.split(".").pop().toLowerCase();
    const allowedExtensions = ["jpg", "jpeg", "png"];
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!allowedExtensions.includes(extension) || (file.type && !allowedTypes.includes(file.type))) {
      alert("Invalid file format. Please upload a JPG, JPEG, or PNG recruitment poster.");
      return;
    }

    // Validate size (16 MB)
    if (file.size > 16 * 1024 * 1024) {
      alert("File size exceeds 16 MB limit.");
      return;
    }

    // Show image preview
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewArea.classList.remove("d-none");
      uploadPrompt.classList.add("d-none");
    };
    reader.readAsDataURL(file);

    // Send to OCR API endpoint
    uploadAndExtractOCR(file);
  }

  function uploadAndExtractOCR(file) {
    ocrLoading.classList.remove("d-none");
    ocrNotice.classList.add("d-none");

    const formData = new FormData();
    formData.append("poster_image", file);

    fetch(dropzone.dataset.ocrUrl || "/api/ocr/extract", {
      method: "POST",
      body: formData
    })
      .then((response) => response.json())
      .then((data) => {
        ocrLoading.classList.add("d-none");
        if (data.success) {
          // Fill hidden file info
          const inputPath = document.getElementById("form-image-path");
          const inputName = document.getElementById("form-image-name");
          const inputEngine = document.getElementById("form-ocr-engine");
          const rawTextArea = document.getElementById("form-raw-text");
          const rawTextPreview = document.getElementById("raw-text-preview");

          if (inputPath) inputPath.value = data.file_path;
          if (inputName) inputName.value = data.file_name;
          if (inputEngine) inputEngine.value = data.ocr_engine;
          if (rawTextArea) rawTextArea.value = data.raw_text || "";
          if (rawTextPreview) rawTextPreview.value = data.raw_text || "";

          // Populate editable fields
          const fields = data.extracted_fields || {};
          populateField("company_name", fields.company_name);
          populateField("job_title", fields.job_title);
          populateField("job_description", fields.job_description || data.raw_text);
          populateField("salary", fields.salary);
          populateField("job_location", fields.job_location);
          populateField("required_skills", fields.required_skills);
          populateField("experience_requirements", fields.experience_requirements);
          populateField("education", fields.education);
          populateField("recruiter_email", fields.recruiter_email);
          populateField("contact_number", fields.contact_number || fields.recruiter_phone);
          populateField("application_url", fields.application_url);
          populateField("official_website_url", fields.official_website_url || fields.official_website);
          populateField("official_careers_url", fields.official_careers_url);

          if (!data.ocr_success && data.ocr_error) {
            ocrNoticeText.textContent = data.ocr_error;
            ocrNotice.classList.remove("d-none");
          }
        } else {
          ocrNoticeText.textContent = data.error || "Could not process image.";
          ocrNotice.classList.remove("d-none");
        }
      })
      .catch((err) => {
        ocrLoading.classList.add("d-none");
        ocrNoticeText.textContent = "Network or server error during OCR extraction: " + err.message;
        ocrNotice.classList.remove("d-none");
      });
  }

  function populateField(id, value) {
    const el = document.getElementById("field_" + id);
    if (el) el.value = value || "";
  }

  function resetUpload() {
    fileInput.value = "";
    previewImg.src = "";
    previewArea.classList.add("d-none");
    uploadPrompt.classList.remove("d-none");
    ocrNotice.classList.add("d-none");
    ocrLoading.classList.add("d-none");

    const inputPath = document.getElementById("form-image-path");
    if (inputPath) inputPath.value = "";
    const rawText = document.getElementById("form-raw-text");
    const rawTextPreview = document.getElementById("raw-text-preview");
    if (rawText) rawText.value = "";
    if (rawTextPreview) rawTextPreview.value = "";
  }
});
