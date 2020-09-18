Осталось только реализовать функцию, которая поможет перекрасить изображение в нужное количество цветов.

### Задание

Реализуйте функцию `recolor(image, n_colors)`. Функция должна принимать изображение в виде numpy массива и количество цветов,
перекрасить каждый пиксель изображения в тот цвет, к которому его отнес метод `k_means` и вернуть новое изображение
в виде массива.

Для сохранения изображения можно сначала создать объект ```Pillow.Image```, с которым мы уже сталкивались в задании **Чтение изображения**. Для этого можно воспользоваться методом ```fromarray```, [создающую](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray) изображение из массива. А затем сохранить его используя ```saveimage```.  


<div class="column" style="float: left;width: 45%;padding: 5px;">
    <img src="superman-batman.png" alt="Исходное изображение" style="width:100%">
    <p style="text-align:center;">Исходное изображение</p>
</div>
<div class="column" style="float: left;width: 45%;padding: 5px;">
    <img src="superman-batman-after.png" alt="16-цветное изображение" style="width:100%">
    <p style="text-align:center;">8-цветное изображение</p>
</div>


