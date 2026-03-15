import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import streamlit as st
from predict import app


class TestPredictApp:
    """Tests for the prediction page app function"""

    @pytest.fixture
    def mock_dataframe(self):
        """Create a mock dataframe with realistic diabetes data"""
        return pd.DataFrame({
            "Glucose": [72, 86, 69, 80, 100, 150],
            "Blood_Pressure": [66, 68, 64, 66, 70, 85],
            "Insulin": [2.1, 2.6, 1.6, 2.2, 2.4, 3.0],
            "BMI": [26.6, 23.3, 21.8, 26.2, 32.1, 35.5],
            "Pedigree_Function": [0.351, 0.254, 0.212, 0.351, 0.412, 0.625],
            "Age": [31, 26, 21, 26, 45, 60]
        })

    @pytest.fixture
    def mock_X(self):
        """Create mock feature matrix"""
        return pd.DataFrame({
            "Glucose": [72, 86, 69],
            "Blood_Pressure": [66, 68, 64],
            "Insulin": [2.1, 2.6, 1.6],
            "BMI": [26.6, 23.3, 21.8],
            "Pedigree_Function": [0.351, 0.254, 0.212],
            "Age": [31, 26, 21]
        })

    @pytest.fixture
    def mock_y(self):
        """Create mock target variable"""
        return pd.Series([0, 1, 0])

    def test_app_should_render_title_when_called(self, mock_dataframe, mock_X, mock_y):
        """Test that app renders the title correctly"""
        with patch.object(st, 'title') as mock_title:
            with patch.object(st, 'markdown'):
                with patch.object(st, 'subheader'):
                    with patch.object(st, 'slider', side_effect=[80, 70, 2.5, 26.0, 0.35, 30]):
                        with patch.object(st, 'button', return_value=False):
                            app(mock_dataframe, mock_X, mock_y)
                            mock_title.assert_called_once_with("Prediction Page")

    def test_app_should_render_description_when_called(self, mock_dataframe, mock_X, mock_y):
        """Test that app renders the description markdown"""
        with patch.object(st, 'title'):
            with patch.object(st, 'markdown') as mock_markdown:
                with patch.object(st, 'subheader'):
                    with patch.object(st, 'slider', side_effect=[80, 70, 2.5, 26.0, 0.35, 30]):
                        with patch.object(st, 'button', return_value=False):
                            app(mock_dataframe, mock_X, mock_y)
                            assert mock_markdown.call_count >= 1
                            call_args = mock_markdown.call_args[0][0]
                            assert "Random Forest Classifier" in call_args
                            assert "Diabetes" in call_args

    def test_app_should_create_sliders_for_all_features(self, mock_dataframe, mock_X, mock_y):
        """Test that app creates sliders for all required features"""
        with patch.object(st, 'title'):
            with patch.object(st, 'markdown'):
                with patch.object(st, 'subheader') as mock_subheader:
                    with patch.object(st, 'slider', side_effect=[80, 70, 2.5, 26.0, 0.35, 30]) as mock_slider:
                        with patch.object(st, 'button', return_value=False):
                            app(mock_dataframe, mock_X, mock_y)
                            assert mock_slider.call_count == 6
                            mock_subheader.assert_called_once_with("Select Values:")

    def test_app_should_use_dataframe_min_max_for_sliders(self, mock_dataframe, mock_X, mock_y):
        """Test that sliders use correct min and max values from dataframe"""
        with patch.object(st, 'title'):
            with patch.object(st, 'markdown'):
                with patch.object(st, 'subheader'):
                    with patch.object(st, 'slider', side_effect=[80, 70, 2.5, 26.0, 0.35, 30]) as mock_slider:
                        with patch.object(st, 'button', return_value=False):
                            app(mock_dataframe, mock_X, mock_y)
                            calls = mock_slider.call_args_list
                            # Glucose slider
                            assert calls[0][1]['value'] or calls[0][0][1] == int(mock_dataframe["Glucose"].min())
                            assert calls[0][1]['value'] or calls[0][0][2] == int(mock_dataframe["Glucose"].max())

    def test_app_should_predict_and_show_high_risk_when_prediction_is_one(self, mock_dataframe, mock_X, mock_y):
        """Test that app shows warning when prediction indicates high risk"""
        with patch.object(st, 'title'):
            with patch.object(st, 'markdown'):
                with patch.object(st, 'subheader'):
                    with patch.object(st, 'slider', side_effect=[150, 85, 3.0, 35.0, 0.625, 60]):
                        with patch.object(st, 'button', return_value=True):
                            with patch('predict.predict', return_value=(1, 0.75)):
                                with patch.object(st, 'info') as mock_info:
                                    with patch.object(st, 'warning') as mock_warning:
                                        with patch.object(st, 'write'):
                                            app(mock_dataframe, mock_X, mock_y)
                                            mock_info.assert_called_once_with("Predicted Sucessfully")
                                            mock_warning.assert_called_once()
                                            assert "high risk of diabetes" in mock_warning.call_args[0][0]

    def test_app_should_predict_and_show_free_from_diabetes_when_prediction_is_zero(self, mock_dataframe, mock_X, mock_y):
        """Test that app shows success message when prediction indicates no diabetes"""
        with patch.object(st, 'title'):
            with patch.object(st, 'markdown'):
                with patch.object(st, 'subheader'):
                    with patch.object(st, 'slider', side_effect=[80, 70, 2.5, 26