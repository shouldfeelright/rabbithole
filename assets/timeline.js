(function () {
  "use strict";

  var el = document.getElementById("timeline-data");
  var viewport = document.getElementById("readme-viewport");
  var slider = document.getElementById("timeline-slider");
  var label = document.getElementById("timeline-change-label");

  if (!el || !viewport || !slider || !label) return;

  var raw = el.textContent.trim();
  var data;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    viewport.innerHTML =
      "<p>Timeline data is missing or invalid. Run <code>python build_timeline.py</code>.</p>";
    return;
  }

  var steps = data.steps || [];
  var max = Math.max(0, steps.length - 1);

  slider.min = "0";
  slider.max = String(max);
  slider.value = String(max);
  slider.setAttribute("aria-valuemin", "0");
  slider.setAttribute("aria-valuemax", String(max));
  slider.setAttribute("aria-valuenow", String(max));

  function describeStep(index) {
    var step = steps[index];
    if (!step) return;
    viewport.innerHTML = step.html || "";

    var meta = step.meta;
    var i = parseInt(index, 10);
    if (!meta || i <= 0) {
      label.innerHTML =
        "<span>Original README — no edits yet. Move the slider right to see each change.</span>";
      slider.setAttribute("aria-valuenow", String(i));
      return;
    }

    var from = meta.from || "";
    var to = meta.to || "";
    label.innerHTML =
      "Change at this step: <strong>" +
      escapeHtml(from) +
      "</strong> → <strong>" +
      escapeHtml(to) +
      "</strong>";
    if (meta.timestamp) {
      label.innerHTML +=
        '<br><span class="timeline-ts">' + escapeHtml(meta.timestamp) + "</span>";
    }

    slider.setAttribute("aria-valuenow", String(i));
    var announcer =
      "Step " + (i + 1) + " of " + steps.length + ". Changed " + from + " to " + to + ".";
    slider.setAttribute("aria-valuetext", announcer);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  slider.addEventListener("input", function () {
    describeStep(slider.value);
  });

  describeStep(slider.value);
})();
