## Setup

Use the following commands to clone the repository, create and activate a Python virtual environment, and pip install all necessary libraries for this web application. All of the requirements for this web application are listed in requirements.txt.

git clone https://github.com/arjunk820/arg-analyzer.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Usage

- Use the command `python app.py` to run the web application
- Go to http://127.0.0.1:5000/ in order to see the web application your local machine.
- Enter an argument in the form's text area and press submit.
- A quality score and a copy of the argument inputted are returned by the application.
- Reload the webpage to enter a new argument in.

## Application

- The app.py file renders the index.html which uses styles.css for aesthetics.
- After the user enters his argument and presses submit, index.html passes the argument on to the app.py
- app.py will call a function which processes the argument entered and use a pre-trained BERT model to give a quality score.
- app.py returns this score with the argument to index.html which then displays this information.
