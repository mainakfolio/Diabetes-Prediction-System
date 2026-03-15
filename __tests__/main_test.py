import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestMainModule:
    """Test suite for main.py"""

    @patch('main.st')
    @patch('main.load_data')
    @patch('main.home')
    @patch('main.data')
    @patch('main.predict')
    @patch('main.visualise')
    @patch('main.about')
    def test_page_configuration_set_correctly(
        self,
        mock_about,
        mock_visualise,
        mock_predict,
        mock_data,
        mock_home,
        mock_load_data,
        mock_st
    ):
        """should set page configuration with correct title and layout"""
        mock_load_data.return_value = (Mock(), Mock(), Mock())
        
        import main
        
        mock_st.set_page_config.assert_called_once()
        call_kwargs = mock_st.set_page_config.call_args[1]
        assert call_kwargs['page_title'] == 'Diabetes Prediction System'
        assert call_kwargs['layout'] == 'wide'

    @patch('main.st')
    @patch('main.load_data')
    @patch('main.home')
    @patch('main.data')
    @patch('main.predict')
    @patch('main.visualise')
    @patch('main.about')
    def test_sidebar_navigation_created(
        self,
        mock_about,
        mock_visualise,
        mock_predict,
        mock_data,
        mock_home,
        mock_load_data,
        mock_st
    ):
        """should create sidebar with navigation title"""
        mock_load_data.return_value = (Mock(), Mock(), Mock())
        
        import main
        
        mock_st.sidebar.title.assert_called_with("Navigation")

    @patch('main.st')
    @patch('main.load_data')
    @patch('main.home')
    @patch('main.data')
    @patch('main.predict')
    @patch('main.visualise')
    @patch('main.about')
    def test_tabs_dictionary_contains_all_pages(
        self,
        mock_about,
        mock_visualise,
        mock_predict,
        mock_data,
        mock_home,
        mock_load_data,
        mock_st
    ):
        """should initialize Tabs dictionary with all page modules"""
        mock_load_data.return_value = (Mock(), Mock(), Mock())
        
        import main
        
        expected_pages = ["Home", "Data Info", "Prediction", "Visualisation", "About me"]
        assert hasattr(main, 'Tabs')
        assert all(page in main.Tabs for page in expected_pages)

    @patch('main.st')
    @patch('main.load_data')
    @patch('main.home')
    @patch('main.data')
    @patch('main.predict')
    @patch('main.visualise')
    @patch('main.about')
    def test_load_data_called_on_startup(
        self,
        mock_about,
        mock_visualise,
        mock_predict,
        mock_data,
        mock_home,
        mock_load_data,
        mock_st
    ):
        """should call load_data function to get dataframe and features"""
        mock_df = Mock()
        mock_X = Mock()
        mock_y = Mock()
        mock_load_data.return_value = (mock_df, mock_X, mock_y)
        
        import main
        
        mock_load_data.assert_called_once()

    @patch('main.st')
    @patch('main.load_data')
    @patch('main.home')
    @patch('main.data')
    @patch('main.predict')
    @patch('main.visualise')
    @patch('main.about')
    def test_home_page_app_called_without_data(
        self,
        mock_about,
        mock_visualise,
        mock_predict,
        mock_data,
        mock_home,
        mock_load_data,
        mock_st
    ):
        """should call home page app function with no arguments"""
        mock_load_data.return_value = (Mock(), Mock(), Mock())
        mock_st.sidebar.radio.return_value = "Home"
        mock_home.app = Mock()
        
        import main
        
        mock_home.app.assert_called_once_with()

    @patch('main.st')
    @patch('main.load_data')
    @patch('main.home')
    @patch('main.data')
    @patch('main.predict')
    @patch('main.visualise')
    @patch('main.about')
    def test_about_page_app_called_without_data(
        self,
        mock_about,
        mock_visualise,
        mock_predict,
        mock_data,
        mock_home,
        mock_load_data,
        mock_st
    ):
        """should call about page app function with no arguments"""
        mock_load_data.return_value = (Mock(), Mock(), Mock())
        mock_st.sidebar.radio.return_value = "About me"
        mock_about.app = Mock()
        
        import main
        
        mock_about.app.assert_called_once_with()

    @patch('main.st')
    @patch('main.load_data')
    @patch('main.home')
    @patch('main.data')
    @patch('main.predict')
    @patch('main.visualise')
    @patch('main.about')
    def test_data_info_page_app_called_with_dataframe(
        self,
        mock_about,
        mock_visualise,
        mock_predict,
        mock_data,
        mock_home,
        mock_load_data,
        mock_st
    ):
        """should call data info page app function with dataframe only"""
        mock_df = Mock()
        mock_X = Mock()
        mock_y = Mock()
        mock_load_data.return_value = (mock_df, mock_X, mock_y)
        mock_st.sidebar.radio.return_value = "Data Info"
        mock_data.app = Mock()
        
        import main
        
        mock_data.app.assert_called_once_with(mock_df)

    @patch('main.st')
    @patch('main.load_data')
    @patch('main.home')
    @patch('main.data')
    @patch('main.predict')
    @patch('main.visualise')
    @patch('main.about')
    def test_prediction_page_app_called_with_all_data(
        self,
        mock_about,
        mock_visualise,
        mock_predict,
        mock_data,
        mock_home,
        mock_load_data,
        mock_st
    ):
        """should call prediction page app function with dataframe and features"""
        mock_df = Mock()
        mock_X = Mock()
        mock_y = Mock()
        mock_load_data.return_value = (mock_df, mock_X, mock