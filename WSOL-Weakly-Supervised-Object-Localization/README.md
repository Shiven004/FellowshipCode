# WSOL-Weakly Supervised Object-Localization

Platform.AI allows domain experts to quickly label large datasets and train deep learning models for multi-class and multi-label classification. It speeds up the labeling process by creating 2D projections that allow the user to label many instances with a single lasso selection.

Users would like to able to also create localization models using Platform.AI however drawing bounding boxes on images is tedious and time consuming. We haven’t figure out how to speed up this process using the projections (perhaps there is a way?). Instead we would like to keep the labeling process the same (ie image level classification labels only) however at inference time produce localization predictions. This means the user continues to label for classification (this is an image of a dog), but the model can predict bounding boxes (the dog appears in this part of the image).

The initial objective of this project is to evaluate various approaches available and identify a generalizable solution that we can add to Platform.AI


After an MVP feature had been added, we would like investigate ways to enhance the initial capability to go beyond public implementations.

Questions / Constraints

How do we evaluate the accuracy of the predicted bounding box? Intersection over Union is one such accuracy score.
How do we handle multiple detections? Perhaps initially we limit to just one detection.
Same exact code / approach should generalize to food, medical, fashion use cases.
What are public benchmarks for this task and how does our approach compare?
What can be added to the Platform.AI user interface to better support this feature?
