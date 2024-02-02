
# Illustrated Character Face Super-Deformation via Unsupervised Image-to-Image Translation

[[Paper](#)] | [[Code](#)]

<div align="center">
 	<img alt="results" src="./assets/deformation_result_v2.png">
</div>

## Abstract

> Super-deformation in character design refers to a simplified modeling of character illustrations that are drawn in detail. Such super-deformation requires both texture and geometrical translation. However, directly adopting conventional image-to-image translation methods for super-deformation is challenging as these methods use a pixel-wise loss which makes the translated images highly dependent on the spatial information of the input image. This study propose a novel deep architecture-based method for the super-deformation of illustrated character faces using an unpaired dataset of detailed and super-deformed character face images collected from the Internet. First, we created a dataset construction pipeline based on image classification and character face detection using deep learning. Then, we designed a generative adversarial network (GAN) that was trained using two discriminators, each for detailed and super-deformed images, and a single generator, capable of synthesizing identical pairs of characters with different textural and geometrical appearance. As ornaments are an important element in character identification, we further introduced ornament augmentation to enable the generator to synthesize a variety of ornaments on the generated character faces. Finally, we constructed a loss function to project character illustrations provided by the user to the learned GAN latent space, which can find an identical super-deformed version. The experimental results show that compared to baseline methods, the proposed method can successfully translate character illustrations to identical super-deformed versions. The codes are available on the Internet.

## Citation

```text
@article{
    authors={Sawada, Tomoya and Katsurai, Marie},
    title={Illustrated Character Face Deformation via Unsupervised Image-to-Image Translation},
    journal={Multimedia System},
    volume={},
    pages={}
}
```

## License

[License](./LICENSE)

## Author

[Tomoya Sawada](https://github.com/STomoya/)
