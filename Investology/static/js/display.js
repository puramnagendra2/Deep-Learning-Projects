document.addEventListener("DOMContentLoaded", function() {
  const searchIcon = document.getElementById("searchIcon");
  const searchBox = document.querySelector(".search-box");

  searchIcon.addEventListener("click", function() {
    // Toggle the 'openSearch' class on the nav element to show/hide the search box
    document.querySelector(".nav").classList.toggle("openSearch");

    // Focus on the search input field when the search box is opened
    if (searchBox.classList.contains("openSearch")) {
      searchBox.querySelector("input").focus();
    }
  });

  // Close the search box when the close button is clicked
  const closeButton = document.querySelector(".nav .navCloseBtn");
  if (closeButton) {
    closeButton.addEventListener("click", function() {
      document.querySelector(".nav").classList.remove("openSearch");
    });
  }
});
