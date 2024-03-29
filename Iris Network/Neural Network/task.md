<html>

<h2>Нейронная сеть</h2>

<p>Нейронная сеть – это совокупность нейронов, соединенных вместе. На самом деле нейросеть – тоже функция, как и единичный нейрон, но часто очень сложная и многопараметрическая. </p>
<p>Нейронные сети состоят из следующих частей:</p>

<ul>
<li>Слой входных нейронов (input layer)</li>
<li>Произвольное количество скрытых нейронов (hidden layers)</li>
<li>Слой выходных нейронов (output layer)</li>
<li>Набор весов и смещений между слоями, $W$</li>
<li>Выбор функции активации для каждого из скрытых слоев $\sigma$</li>
</ul>

<p>Вес &mdash; единственный параметр связи между двумя нейронами. Благодаря весам, входная информация изменяется, когда передается от одного нейрона к другому.</p>

<p>Исторически при инициализации весов использовались небольшие случайные числа (в этой задаче тоже будет так), но в последнее время были разработаны 
эвристики, принимающие во внимание такую информацию, как тип используемой функции активации и количество входов в узел: такие более адаптированные 
подходы позволяют увеличить эффективность обучения нейронных сетей.</p>

<p>Ниже представлена 2-слойная нейронная сеть. При подсчете количества слоев слой входных нейронов не учитывается.</p>

<figure>
  <img src="neuralnet.png" alt="NeuralNet" style="width:100%">
</figure>

<p>При первом запуске нейросети ответ будет далек от правильного, поскольку сеть не обучена. Чтобы улучшить результат
(то есть то, насколько результат классификации тестовой выборки соответствует реальным классам, к
которым относятся объекты в ней), необходимо произвести обучение. 
О том, как это происходит, мы расскажем в следующих шагах этого урока.</p>



</html>