document.addEventListener('DOMContentLoaded', () => {
  const menu = document.querySelector('.menu-toggle');
  const sidebar = document.querySelector('.sidebar');
  menu?.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    menu.setAttribute('aria-expanded', String(open));
  });
  document.querySelectorAll('[data-confirm]').forEach(form => {
    form.addEventListener('submit', event => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });
  const file = document.querySelector('#training-file');
  file?.addEventListener('change', () => {
    document.querySelector('#selected-file').textContent = file.files[0]?.name || '';
  });
  document.querySelectorAll('.matrix-toggle').forEach(button => {
    button.addEventListener('click', () => {
      const row = document.getElementById(button.dataset.matrix);
      row.hidden = !row.hidden;
      button.setAttribute('aria-expanded', String(!row.hidden));
      button.textContent = row.hidden ? 'Details ↓' : 'Hide ↑';
    });
  });
});
