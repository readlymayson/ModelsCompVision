// Dataset Viewer JavaScript

// Глобальные переменные
let currentPage = 1;
let currentSearch = '';
let currentClassFilter = '';
let totalPages = 1;
let totalImages = 0;
let classNames = {};
let isLoading = false;

// Инициализация галереи
async function initGallery() {
    try {
        // Загрузить статистику для получения имен классов
        const statsResponse = await fetch('/api/stats');
        const stats = await statsResponse.json();
        classNames = stats.class_names || {};

        // Заполнить фильтр классов
        populateClassFilter(stats.class_distribution || []);

        // Загрузить первую страницу изображений
        await loadImages(1, '', '');
    } catch (error) {
        console.error('Ошибка инициализации галереи:', error);
        showError('Ошибка загрузки галереи');
    }
}

// Заполнение фильтра классов
function populateClassFilter(classDistribution) {
    const classFilter = document.getElementById('class-filter');

    // Очистить существующие опции кроме "Все классы"
    while (classFilter.children.length > 1) {
        classFilter.removeChild(classFilter.lastChild);
    }

    // Добавить классы
    classDistribution.forEach(classInfo => {
        const option = document.createElement('option');
        option.value = classInfo.id;
        option.textContent = `${classInfo.name} (${classInfo.count})`;
        classFilter.appendChild(option);
    });
}

// Загрузка изображений с пагинацией
async function loadImages(page = 1, search = '', classFilter = '') {
    if (isLoading) return; // Предотвратить множественные одновременные запросы

    const gallery = document.getElementById('gallery');
    const loading = document.getElementById('loading');
    const pagination = document.getElementById('pagination');

    isLoading = true;

    // Показать загрузку
    loading.style.display = 'block';

    // Всегда очищать галерею при загрузке новых изображений
    gallery.innerHTML = '';

    try {
        // Сначала получить общее количество изображений
        const countParams = new URLSearchParams({
            search: search,
            class_filter: classFilter
        });

        const countResponse = await fetch(`/api/images/count?${countParams}`);
        const countData = await countResponse.json();

        if (countData.error) {
            throw new Error(countData.error);
        }

        totalImages = countData.count;
        totalPages = Math.ceil(totalImages / 50);

        // Теперь загрузить изображения для текущей страницы
        const params = new URLSearchParams({
            page: page,
            per_page: 50,
            search: search,
            class_filter: classFilter
        });

        const response = await fetch(`/api/images?${params}`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Скрыть загрузку
        loading.style.display = 'none';

        // Обновить глобальные переменные
        currentPage = data.page;
        currentSearch = search;
        currentClassFilter = classFilter;

        // Отобразить изображения
        displayImages(data.images);

        // Отобразить пагинацию
        displayPagination(data.page, totalPages, totalImages);

    } catch (error) {
        console.error('Ошибка загрузки изображений:', error);
        loading.style.display = 'none';
        showError('Ошибка загрузки изображений');
    } finally {
        isLoading = false;
    }
}

// Отображение изображений
function displayImages(images, replace = true) {
    const gallery = document.getElementById('gallery');

    if (replace) {
        gallery.innerHTML = '';
    }

    images.forEach(image => {
        const imageCard = createImageCard(image);
        gallery.appendChild(imageCard);
    });
}

// Создание карточки изображения
function createImageCard(image) {
    const card = document.createElement('div');
    card.className = 'image-card';
    card.onclick = () => window.location.href = `/image/${encodeURIComponent(image.path)}`;

    const img = document.createElement('img');
    img.src = image.thumbnail || '/static/placeholder.png';
    img.alt = image.name;
    img.loading = 'lazy';

    const info = document.createElement('div');
    info.className = 'image-info';

    const name = document.createElement('div');
    name.className = 'image-name';
    name.textContent = image.name;

    const stats = document.createElement('div');
    stats.className = 'image-stats';
    stats.textContent = `${image.annotations_count} аннотаций`;

    if (image.classes && image.classes.length > 0) {
        const classesText = image.classes.map(classId =>
            classNames[classId] || `class_${classId}`
        ).join(', ');
        stats.textContent += ` | Классы: ${classesText}`;
    }

    info.appendChild(name);
    info.appendChild(stats);

    card.appendChild(img);
    card.appendChild(info);

    return card;
}

// Отображение пагинации
function displayPagination(currentPage, totalPages, totalImages) {
    const pagination = document.getElementById('pagination');
    pagination.innerHTML = '';

    if (totalPages <= 1) return;

    // Предыдущая страница
    if (currentPage > 1) {
        const prevBtn = createPageButton('←', currentPage - 1);
        pagination.appendChild(prevBtn);
    }

    // Номера страниц
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);

    if (startPage > 1) {
        const firstBtn = createPageButton('1', 1);
        pagination.appendChild(firstBtn);
        if (startPage > 2) {
            const dots = document.createElement('span');
            dots.textContent = '...';
            dots.style.margin = '0 0.5rem';
            pagination.appendChild(dots);
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        const pageBtn = createPageButton(i.toString(), i);
        if (i === currentPage) {
            pageBtn.classList.add('active');
        }
        pagination.appendChild(pageBtn);
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            const dots = document.createElement('span');
            dots.textContent = '...';
            dots.style.margin = '0 0.5rem';
            pagination.appendChild(dots);
        }
        const lastBtn = createPageButton(totalPages.toString(), totalPages);
        pagination.appendChild(lastBtn);
    }

    // Следующая страница
    if (currentPage < totalPages) {
        const nextBtn = createPageButton('→', currentPage + 1);
        pagination.appendChild(nextBtn);
    }

    // Информация об общем количестве
    const info = document.createElement('div');
    info.textContent = `Всего изображений: ${totalImages}`;
    info.style.marginTop = '1rem';
    info.style.textAlign = 'center';
    info.style.color = '#666';
    pagination.appendChild(info);
}

// Создание кнопки страницы
function createPageButton(text, page) {
    const btn = document.createElement('button');
    btn.className = 'page-btn';
    btn.textContent = text;
    btn.onclick = () => loadImages(page, currentSearch, currentClassFilter);
    return btn;
}

// Обработчики событий для поиска и фильтрации
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const classFilter = document.getElementById('class-filter');
    const clearFiltersBtn = document.getElementById('clear-filters-btn');
    const clearCacheBtn = document.getElementById('clear-cache-btn');

    if (searchBtn) {
        searchBtn.onclick = function() {
            const newSearch = searchInput.value.trim();
            if (newSearch !== currentSearch) {
                currentSearch = newSearch;
                currentPage = 1;
                loadImages(currentPage, currentSearch, currentClassFilter);
            }
        };

        searchInput.onkeypress = function(e) {
            if (e.key === 'Enter') {
                searchBtn.click();
            }
        };
    }

    if (classFilter) {
        classFilter.onchange = function() {
            const newClassFilter = classFilter.value;
            if (newClassFilter !== currentClassFilter) {
                currentClassFilter = newClassFilter;
                currentPage = 1;
                loadImages(currentPage, currentSearch, currentClassFilter);
            }
        };
    }

    if (clearFiltersBtn) {
        clearFiltersBtn.onclick = function() {
            if (currentSearch || currentClassFilter) {
                searchInput.value = '';
                classFilter.value = '';
                currentSearch = '';
                currentClassFilter = '';
                currentPage = 1;
                loadImages(currentPage, currentSearch, currentClassFilter);
            }
        };
    }

    if (clearCacheBtn) {
        clearCacheBtn.onclick = async function() {
            try {
                const response = await fetch('/api/cache/clear', { method: 'POST' });
                const data = await response.json();
                if (data.message) {
                    // Перезагрузить текущую страницу
                    loadImages(currentPage, currentSearch, currentClassFilter);
                    showMessage('Кэш очищен');
                } else if (data.error) {
                    throw new Error(data.error);
                }
            } catch (error) {
                console.error('Ошибка очистки кэша:', error);
                showError('Ошибка очистки кэша');
            }
        };
    }
});

// Инициализация просмотра изображения
async function initImageViewer(imageName) {
    const mainImage = document.getElementById('main-image');
    const imageNameElement = document.getElementById('image-name');
    const loading = document.getElementById('loading');

    try {
        // Установить имя файла
        imageNameElement.textContent = imageName;

        // Загрузить изображение с визуализацией
        mainImage.src = `/api/visualize/${imageName}`;
        mainImage.style.display = 'none';
        loading.style.display = 'block';

        // Дождаться загрузки изображения
        mainImage.onload = function() {
            loading.style.display = 'none';
            mainImage.style.display = 'block';
        };

        mainImage.onerror = function() {
            loading.style.display = 'none';
            console.error('Ошибка загрузки изображения:', mainImage.src);
            showError('Ошибка загрузки изображения. Проверьте путь к файлу.');
        };

        // Загрузить аннотации
        await loadAnnotations(imageName);

        // Настроить кнопки навигации
        setupNavigation(imageName);

    } catch (error) {
        console.error('Ошибка инициализации просмотра изображения:', error);
        showError('Ошибка загрузки изображения');
    }
}

// Загрузка аннотаций
async function loadAnnotations(imageName) {
    const annotationsList = document.getElementById('annotations-list');
    const loading = document.getElementById('annotations-loading');

    try {
        const response = await fetch(`/api/annotations/${imageName}`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        loading.style.display = 'none';

        if (data.annotations.length === 0) {
            annotationsList.innerHTML = '<p>Аннотаций нет</p>';
            return;
        }

        annotationsList.innerHTML = '';
        data.annotations.forEach((annotation, index) => {
            const item = createAnnotationItem(annotation, index);
            annotationsList.appendChild(item);
        });

    } catch (error) {
        console.error('Ошибка загрузки аннотаций:', error);
        loading.style.display = 'none';
        annotationsList.innerHTML = '<p class="error">Ошибка загрузки аннотаций</p>';
    }
}

// Создание элемента аннотации
function createAnnotationItem(annotation, index) {
    const item = document.createElement('div');
    item.className = 'annotation-item';

    const classDiv = document.createElement('div');
    classDiv.className = 'annotation-class';
    classDiv.textContent = `${index + 1}. ${annotation.class_name}`;

    const coordsDiv = document.createElement('div');
    coordsDiv.className = 'annotation-coords';
    coordsDiv.textContent = `x: ${annotation.x_center.toFixed(3)}, y: ${annotation.y_center.toFixed(3)}, w: ${annotation.width.toFixed(3)}, h: ${annotation.height.toFixed(3)}`;

    item.appendChild(classDiv);
    item.appendChild(coordsDiv);

    return item;
}

// Настройка навигации между изображениями
function setupNavigation(currentImageName) {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const toggleBtn = document.getElementById('toggle-annotations-btn');

    let showAnnotations = true;
    let originalImageSrc = `/api/visualize/${currentImageName}`;
    let plainImageSrc = `/api/image/${currentImageName}`;

    // Кнопка переключения аннотаций
    if (toggleBtn) {
        toggleBtn.onclick = function() {
            const mainImage = document.getElementById('main-image');
            showAnnotations = !showAnnotations;
            mainImage.src = showAnnotations ? originalImageSrc : plainImageSrc;
            toggleBtn.textContent = showAnnotations ? 'Скрыть аннотации' : 'Показать аннотации';
        };
    }

    // Заглушки для навигации (пока без реализации)
    if (prevBtn) {
        prevBtn.onclick = function() {
            alert('Навигация между изображениями будет реализована в следующей версии');
        };
    }

    if (nextBtn) {
        nextBtn.onclick = function() {
            alert('Навигация между изображениями будет реализована в следующей версии');
        };
    }
}

// Инициализация статистики
async function initStats() {
    const loading = document.getElementById('loading');
    const content = document.getElementById('stats-content');

    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        if (stats.error) {
            throw new Error(stats.error);
        }

        loading.style.display = 'none';
        content.style.display = 'block';

        // Заполнить общую статистику
        document.getElementById('total-images').textContent = stats.total_images || 0;
        document.getElementById('valid-images').textContent = stats.valid_images || 0;
        document.getElementById('total-annotations').textContent = stats.total_annotations || 0;
        document.getElementById('missing-labels').textContent = stats.missing_labels || 0;
        document.getElementById('empty-labels').textContent = stats.empty_labels || 0;
        document.getElementById('annotation-errors').textContent = stats.annotation_errors || 0;

        // Отобразить распределение классов
        displayClassDistribution(stats.class_distribution || []);

    } catch (error) {
        console.error('Ошибка загрузки статистики:', error);
        loading.style.display = 'none';
        showError('Ошибка загрузки статистики');
    }
}

// Отображение распределения классов
function displayClassDistribution(classDistribution) {
    const container = document.getElementById('class-distribution');
    const loading = document.getElementById('class-loading');

    loading.style.display = 'none';

    if (classDistribution.length === 0) {
        container.innerHTML = '<p>Классов не найдено</p>';
        return;
    }

    const chart = document.createElement('div');
    chart.className = 'class-chart';

    classDistribution.forEach((classInfo, index) => {
        const item = document.createElement('div');
        item.className = 'class-item';

        const color = document.createElement('div');
        color.className = 'class-color';
        color.style.backgroundColor = getClassColor(index);

        const name = document.createElement('span');
        name.textContent = classInfo.name;

        const count = document.createElement('span');
        count.textContent = classInfo.count;
        count.style.fontWeight = 'bold';

        item.appendChild(color);
        item.appendChild(name);
        item.appendChild(count);

        chart.appendChild(item);
    });

    container.appendChild(chart);
}

// Получение цвета для класса
function getClassColor(index) {
    const colors = [
        '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
        '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f'
    ];
    return colors[index % colors.length];
}

// Вспомогательные функции
function showError(message) {
    showNotification(message, 'error');
}

function showMessage(message) {
    showNotification(message, 'success');
}

function showNotification(message, type = 'error') {
    // Создать элемент уведомления
    const notificationDiv = document.createElement('div');
    notificationDiv.className = `notification ${type}`;
    notificationDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#e74c3c' : '#27ae60'};
        color: white;
        padding: 1rem;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        z-index: 1000;
        max-width: 300px;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
    `;
    notificationDiv.textContent = message;

    document.body.appendChild(notificationDiv);

    // Показать уведомление
    setTimeout(() => {
        notificationDiv.style.opacity = '1';
        notificationDiv.style.transform = 'translateX(0)';
    }, 10);

    // Удалить через 5 секунд
    setTimeout(() => {
        notificationDiv.style.opacity = '0';
        notificationDiv.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notificationDiv.parentNode) {
                notificationDiv.parentNode.removeChild(notificationDiv);
            }
        }, 300);
    }, 5000);
}
