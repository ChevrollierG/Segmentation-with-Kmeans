# Segmentation-with-Kmeans

### The project

This project consists on an implementation of the clustering algorithm K-means from scratch to perform segmentation with classical machine learning.

We use OpenCV to load images and treat every pixels individually compared to others on 7 attributes:  
-R, the red color in the RGB code  
-G, the green color in the RGB code  
-B, the blue color in the RGB code  
-X, the horizontal coordinate in the image  
-Y, the vertical coordinate in the image  
-edge_x, horizontal value of the edge calculated with the other pixels and the sobel operator  
-edge_y, vertical value of the edge calculated with the other pixels and the sobel operator  

I ran the algorithm on different examples and decided to put weights on each pixel's attributes, so 0.1 for the coordinates, the edges and 0.4 for the colors.

At the beginning, the image is plotted with matplotlib, then we run the algorithm, plot the result image with colored classes and each classes separately with it's true color.

### Conclusion

Practicing segmentation is one of the most difficult exercise in computer vision and doing it with K-means allow us to have a first step with it. Indeed, it performs segmentation but it's not perfect as we use clustering, the classes are not labelled as if we trained a neural network on a labelled dataset and some very spatially close different objects with almost the same color end up with the same class predicted

### Results

Input  
![image](https://user-images.githubusercontent.com/91634314/184505162-087bb404-5092-4379-b82a-5aec2321245f.png)

Output  
![image](https://user-images.githubusercontent.com/91634314/184505177-ab9c3b44-82aa-4e47-a7bd-49143aa3d4c0.png)


### Documentation

https://web.eecs.umich.edu/~jjcorso/t/598F14/files/lecture_1006_clustering.pdf
