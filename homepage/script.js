"use strict";

(() => {
  const workflow = document.getElementById("workflow");
  const toggle = document.getElementById("animation-toggle");
  const label = document.getElementById("animation-label");
  const icon = toggle.querySelector(".pause-icon");
  const description = document.getElementById("cycle-description");
  const steps = [...document.querySelectorAll(".cycle-step")];
  const indicators = [...document.querySelectorAll(".cycle-indicator i")];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const descriptions = [
    "New assay outcomes join the feedback set.",
    "The fixed editor estimates and validates a parameter correction.",
    "Accepted changes are saved into the predictor's native parameters.",
    "The updated predictor chooses the next experimental batch."
  ];
  let activeStep = 0;
  let paused = reducedMotion.matches;
  let timer = null;

  function renderStep() {
    steps.forEach((step, i) => step.classList.toggle("is-active", i === activeStep));
    indicators.forEach((dot, i) => dot.classList.toggle("is-active", i === activeStep));
    description.textContent = descriptions[activeStep];
  }

  function setPaused(value) {
    paused = value;
    workflow.classList.toggle("is-paused", paused);
    toggle.setAttribute("aria-pressed", String(paused));
    label.textContent = paused ? "Play animation" : "Pause animation";
    icon.textContent = paused ? "▷" : "Ⅱ";
    window.clearInterval(timer);
    timer = null;
    if (!paused) timer = window.setInterval(() => {
      activeStep = (activeStep + 1) % steps.length;
      renderStep();
    }, 2800);
  }

  toggle.addEventListener("click", () => setPaused(!paused));
  reducedMotion.addEventListener("change", event => setPaused(event.matches));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      window.clearInterval(timer);
      timer = null;
    } else setPaused(paused);
  });
  renderStep();
  setPaused(paused);

  const status = document.getElementById("copy-status");
  let statusTimer;
  function showStatus(message) {
    status.textContent = message;
    status.classList.add("visible");
    window.clearTimeout(statusTimer);
    statusTimer = window.setTimeout(() => status.classList.remove("visible"), 2200);
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const temporary = document.createElement("textarea");
    temporary.value = text;
    temporary.setAttribute("readonly", "");
    temporary.style.position = "fixed";
    temporary.style.opacity = "0";
    document.body.appendChild(temporary);
    temporary.select();
    const copied = document.execCommand("copy");
    temporary.remove();
    if (!copied) throw new Error("Clipboard unavailable");
  }

  document.querySelectorAll("[data-copy]").forEach(button => {
    button.addEventListener("click", async () => {
      const block = document.getElementById(button.dataset.copy);
      if (!block) return;
      const initialLabel = button.textContent;
      try {
        await copyText(block.textContent.trim());
        button.textContent = "Copied";
        showStatus(button.dataset.copy === "bibtex" ? "BibTeX copied" : "Quickstart commands copied");
        window.setTimeout(() => { button.textContent = initialLabel; }, 1800);
      } catch (_) {
        showStatus("Select the text to copy it.");
      }
    });
  });
})();
