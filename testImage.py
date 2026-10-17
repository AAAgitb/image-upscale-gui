from PIL import ImageChops, Image

#for testing
def testImage(img1, img2, save):
    if(img1.size != img2.size):
        return False

    diff = ImageChops.difference(img1, img2)
    
    if diff.getbbox() is None:
        print("Images are exactly identical.")
        return True
    else:
        print("Images are different.")
        # Optional: Save the difference map to visually see changes
        if(save):
            diff.save("difference_output.png")
        
        return False

    