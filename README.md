# Python Project Readme

This readme provides instructions for setting up and running the Python project. Follow these steps to activate the
virtual environment, install the project's dependencies, and download the necessary CSV big file.

## Getting Started

1. **Clone the Repository:**

   Clone this repository to your local machine using the following command:

````commandline
git clone https://github.com/luisgsilva950/research-optimization-nsgaII-lab4.git
````

2. **Navigate to the Project Directory:**

The necessary files are in the root.

## Setting Up the Virtual Environment

3. **Create a Virtual Environment:**

If you haven't already installed `virtualenv`, you can do so with the following command:

4. **Create a Virtual Environment:**

Create a virtual environment for the project using the following command:

This will create a virtual environment named `venv` in your project folder.

5. **Activate the Virtual Environment:**

Activate the virtual environment based on your operating system:

- On Windows:

  ```
  venv\Scripts\activate
  ```

- On macOS and Linux:

  ```
  source venv/bin/activate
  ```

## Installing Dependencies

6. **Install Project Dependencies:**

Use `pip` to install the project dependencies listed in the `requirements.txt` file:

```commandline
pip install -r requirements.txt
```

This will install all the necessary packages and libraries required for the project.

## Downloading the CSV Big File

7. **Download the CSV Big File:**

You need to download the "customer_to_point_distances.json" CSV big file from the provided source or location. Save the
file in the project directory.

## Running the Project

````commandline
python3 nsga_II.py
````

