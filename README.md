# SageMaker Application Example
Install required packages with ```pip install -r requirements.txt```

Run the script with ```python app.py```

The application loads images from ```mnist.pkl.gz```, picks one example image from the
data set, converts the image to csv format, sends the csv to a SageMaker endpoint,
prints the returning classification and prints the example image.
