
# Face Detection model

The frontal face detection model.

## Usage

### CLI

- Install [detectron2](https://github.com/facebookresearch/detectron2).

- Download weights from [here](#).

- Then on your terminal:

    ```terminal
    python3 frontal_face_detector.py <input> <weights>
    ```

    Replace `<input>` and `<weights>` according to your environment.

#### Options

```terminal
$ python frontal_face_detector.py --help
usage: frontal_face_detector.py [-h] [--threshold THRESHOLD]
                                [--min-size MIN_SIZE] [--output OUTPUT]
                                [--param-file PARAM_FILE] [--quiet]
                                input weights

positional arguments:
input                      Input. Either a file or folder of image file.
weights                    weights of the Faster R-CNN.

optional arguments:
-h, --help                 show this help message and exit
--threshold THRESHOLD      Threshold
--min-size MIN_SIZE        Minimum size of the cropped images.
--output OUTPUT            Folder to save cropped faces.
--param-file PARAM_FILE    If given, save parameters used to crop faces to this file
--quiet                    No verbose.
```


## citations

[Here](../README.md#citation)

If you feel this work is helpful, don't forget to cite these works as well.

```text
@inproceedings{ren_fasterrcnn_2015,
    author = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
    title = {Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
    booktitle = {Advances in Neural Information Processing Systems},
    pages = {91--99},
    volume = {28},
    year = {2015}
}
```

```text
@inproceedings{zheng_icartoon_2020,
    author={Zheng, Yi and Zhao, Yifan and Ren, Mengyuan and Yan, He and Lu, Xiangju and Liu, Junhui and Li, Jia},
    title={Cartoon Face Recognition: A Benchmark Dataset},
    booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
    pages={2264--2272},
    year={2020}
}
```

```text
@misc{wu_detectron2_2019,
  author = {Wu, Yuxin and Kirillov, Alexander and Massa, Francisco and Lo, Wan-Yen and Girshick, Ross},
  title = {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year = {2019}
}
```

## Author

[Tomoya Sawada](https://github.com/STomoya/)
