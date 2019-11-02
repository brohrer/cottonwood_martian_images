# Image Compression using Autoencoders

### Getting started

To run the demo, just clone the repository and run the top-level script.

```bash
git clone https://github.com/brohrer/cottonwood_martian_images.git
cd cottonwood_martian_images
python3 build_image_patches.py
```


#### Cottonwood compatibility

This project makes use of the Cottonwood machine learning framework,
specifically version 7. If you don't already have it, here's
the command to install it.

```bash
python3 -m pip install "cottonwood==7" --user
```

If you have a layer version installed you may
have to uninstall it first, then install version 7,
since backward compatibility is not guaranteed.

```bash
python3 -m pip uninstall cottonwood
```

Then when you're done, you can reinstall the latest version of
Cottonwood with

```bash
python3 -m pip install cottonwood --user --upgrade
```

