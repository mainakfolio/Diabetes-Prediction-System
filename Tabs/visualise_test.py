import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import streamlit as st
from visualise import app


class TestVisualisationApp:
    """Test suite for the visualise module"""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample diabetes dataset for testing"""
        data = {
            'Pregnancies': [6, 1, 8, 1, 0],
            'Glucose': [148, 85, 183, 89, 137],
            'Blood_Pressure': [72, 66, 64, 66, 40],
            'SkinThickness': [35, 29, 0, 23, 35],
            'Insulin': [0, 0, 0, 94, 168],
            'BMI': [33.6, 26.6, 23.3, 28.1, 43.1],
            'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288],
            'Age': [50, 31, 32, 21, 33],
            'Outcome': [1, 0, 1, 0, 1]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_X(self):
        """Create sample feature data"""
        data = {
            'Pregnancies': [6, 1, 8, 1, 0],
            'Glucose': [148, 85, 183, 89, 137],
            'Blood_Pressure': [72, 66, 64, 66, 40],
            'SkinThickness': [35, 29, 0, 23, 35],
            'Insulin': [0, 0, 0, 94, 168],
            'BMI': [33.6, 26.6, 23.3, 28.1, 43.1],
            'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288],
            'Age': [50, 31, 32, 21, 33]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_y(self):
        """Create sample target data"""
        return pd.Series([1, 0, 1, 0, 1], name='Outcome')

    @patch('streamlit.set_option')
    @patch('streamlit.title')
    @patch('streamlit.checkbox')
    def test_app_initializes_correctly(self, mock_checkbox, mock_title, mock_set_option, sample_dataframe, sample_X, sample_y):
        """should initialize app with correct title and settings"""
        mock_checkbox.return_value = False
        
        app(sample_dataframe, sample_X, sample_y)
        
        mock_set_option.assert_called_once()
        mock_title.assert_called_once_with("Visualise the Diabetes Prediction Web app")

    @patch('streamlit.set_option')
    @patch('streamlit.title')
    @patch('streamlit.subheader')
    @patch('streamlit.checkbox')
    @patch('streamlit.pyplot')
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.heatmap')
    def test_app_shows_correlation_heatmap_when_checked(self, mock_heatmap, mock_fig, mock_pyplot, 
                                                         mock_checkbox, mock_subheader, mock_title, 
                                                         mock_set_option, sample_dataframe, sample_X, sample_y):
        """should display correlation heatmap when checkbox is selected"""
        mock_checkbox.side_effect = [True, False, False, False]
        mock_ax = MagicMock()
        mock_ax.get_ylim.return_value = (0, 10)
        mock_heatmap.return_value = mock_ax
        
        app(sample_dataframe, sample_X, sample_y)
        
        mock_subheader.assert_called_once_with("Correlation Heatmap")
        mock_heatmap.assert_called_once()
        mock_pyplot.assert_called()

    @patch('streamlit.set_option')
    @patch('streamlit.title')
    @patch('streamlit.checkbox')
    @patch('streamlit.pyplot')
    @patch('seaborn.color_palette')
    @patch('seaborn.scatterplot')
    def test_app_shows_scatter_plot_when_checked(self, mock_scatter, mock_color_palette, mock_pyplot,
                                                  mock_checkbox, mock_title, mock_set_option,
                                                  sample_dataframe, sample_X, sample_y):
        """should display scatter plot when checkbox is selected"""
        mock_checkbox.side_effect = [False, True, False, False]
        mock_scatter.return_value = MagicMock()
        
        app(sample_dataframe, sample_X, sample_y)
        
        mock_color_palette.assert_called()
        mock_scatter.assert_called_once()
        mock_pyplot.assert_called()

    @patch('streamlit.set_option')
    @patch('streamlit.title')
    @patch('streamlit.checkbox')
    @patch('streamlit.pyplot')
    @patch('seaborn.color_palette')
    @patch('seaborn.histplot')
    def test_app_shows_histogram_when_checked(self, mock_hist, mock_color_palette, mock_pyplot,
                                              mock_checkbox, mock_title, mock_set_option,
                                              sample_dataframe, sample_X, sample_y):
        """should display histogram when checkbox is selected"""
        mock_checkbox.side_effect = [False, False, True, False]
        mock_hist.return_value = MagicMock()
        
        app(sample_dataframe, sample_X, sample_y)
        
        mock_color_palette.assert_called()
        mock_hist.assert_called_once()
        mock_pyplot.assert_called()

    @patch('streamlit.set_option')
    @patch('streamlit.title')
    @patch('streamlit.checkbox')
    @patch('streamlit.graphviz_chart')
    @patch('web_functions.train_model')
    @patch('sklearn.tree.export_graphviz')
    def test_app_shows_decision_tree_when_checked(self, mock_export_graphviz, mock_train_model,
                                                   mock_graphviz_chart, mock_checkbox, mock_title,
                                                   mock_set_option, sample_dataframe, sample_X, sample_y):
        """should display decision tree when checkbox is selected"""
        mock_checkbox.side_effect = [False, False, False, True]
        mock_model = MagicMock()
        mock_train_model.return_value = (mock_model, 0.85)
        mock_export_graphviz.return_value = "digraph { }"
        
        app(sample_dataframe, sample_X, sample_y)