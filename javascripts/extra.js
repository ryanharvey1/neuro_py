// Make the header site title a clickable link to home
document.addEventListener("DOMContentLoaded", function () {
  var topic = document.querySelector(".md-header__topic:first-child .md-ellipsis");
  if (topic && !topic.closest("a")) {
    var link = document.createElement("a");
    // Derive base URL from the logo link (always points home) or fall back to "."
    var logo = document.querySelector("a.md-header__button.md-logo");
    link.href = logo ? logo.getAttribute("href") : ".";
    link.style.color = "inherit";
    link.style.textDecoration = "none";
    link.style.cursor = "pointer";
    while (topic.firstChild) link.appendChild(topic.firstChild);
    topic.appendChild(link);
  }
});
