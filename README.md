# face-segmentation by Ковалёв Александр Евгеньевич

## Последовательное описание решения

### Подготовка данных 
1. Был использован датасет [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ)
2. `prepropess_data.py` был взят из репозитория, указанного в ТЗ, затем переписан так, чтобы лицо получало значение 255, а все остальные объекты получали значение 0.
    Используйте этот скрипт по форме, чтобы запустить генерацию масок в установленном формате: 
```
python preprocess_data.py --masks_parts <your path> --output_mask <your path>
```

### Обучение
Из-за отсутсвия requirements.txt файла в репозитории в ТЗ запуск обучения оказался крайне мучительным, поэтому было принято решение
самостоятельно написать модель для сегментации изображения. За основу архитектуры был взят ResNet50.
    Подробнее с архитектурой вы можете ознакомиться в файле `train.py`.
    Скрипт для запуска обучения:
```
python train.py --batch_size <number> --faces_dir <your path> --masks_dir <your path> --fahand_dir <your path> --mahand_dir <your path> --weights_dir <your path>
```
Если 4 из этих параметра весьма понятны, то fahand_dir и mahand_dir могут оставаться загадкой.
Чтобы понять их предназначение перейдем к следующему этапу.

### Аугментация
После завершения обучения было выявлено, что рука также распознается как лицо. 
<p align="center">
	<img src="./examples/Before_aug.png" alt="Original Input">
</p>
В попытках решить данную проблему было принято решение использовать кусочек EgoHands dataset с кропнутыми руками и наложенными масками 
<table>

<tr>
<th>&nbsp;</th>
<th>Hand</th>
<th>Mask</th>
</tr>

<tr>
<td><em>Example</em></td>
<td><img src="./examples/hand_no_mask.jpg" alt="Original Input"></td>
<td><img src="./examples/hand_masked.png" alt="Original Input"></td>
</tr>

</table>