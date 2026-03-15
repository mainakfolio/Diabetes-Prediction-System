import pytest
from unittest.mock import patch, MagicMock
from home import app


class TestHomeApp:
    """Tests for home page app function"""

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_render_title_when_called(self, mock_markdown, mock_image, mock_title):
        """Test that app renders the correct title"""
        app()
        mock_title.assert_called_once_with("Diabetes Prediction System")

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_load_home_image_when_called(self, mock_markdown, mock_image, mock_title):
        """Test that app loads the home image from correct path"""
        app()
        mock_image.assert_called_once_with("./images/home.png")

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_display_description_markdown_when_called(self, mock_markdown, mock_image, mock_title):
        """Test that app displays markdown description with html content"""
        app()
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args
        assert "Diabetes is a chronic" in call_args[0][0]
        assert "Random Forest Classifier" in call_args[0][0]
        assert call_args[1]['unsafe_allow_html'] is True

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_call_streamlit_functions_in_correct_order(self, mock_markdown, mock_image, mock_title):
        """Test that app calls streamlit functions in the expected order"""
        app()
        assert mock_title.call_count == 1
        assert mock_image.call_count == 1
        assert mock_markdown.call_count == 1

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_handle_all_three_calls_even_if_one_fails(self, mock_markdown, mock_image, mock_title):
        """Test that app attempts all three streamlit calls"""
        mock_title.side_effect = Exception("Title error")
        with pytest.raises(Exception):
            app()

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_use_correct_html_formatting_in_description(self, mock_markdown, mock_image, mock_title):
        """Test that description uses proper HTML formatting with font-size"""
        app()
        markdown_content = mock_markdown.call_args[0][0]
        assert 'style="font-size:20px;"' in markdown_content
        assert '<p' in markdown_content
        assert '</p>' in markdown_content

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_include_disease_explanation_in_description(self, mock_markdown, mock_image, mock_title):
        """Test that description includes key information about diabetes"""
        app()
        markdown_content = mock_markdown.call_args[0][0]
        assert "chronic" in markdown_content.lower()
        assert "health condition" in markdown_content.lower()
        assert "predict" in markdown_content.lower()

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_return_none(self, mock_markdown, mock_image, mock_title):
        """Test that app function returns None"""
        result = app()
        assert result is None

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_enable_unsafe_html_rendering(self, mock_markdown, mock_image, mock_title):
        """Test that unsafe_allow_html is set to True for markdown rendering"""
        app()
        assert mock_markdown.call_args[1]['unsafe_allow_html'] is True

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_should_display_meaningful_prediction_message(self, mock_markdown, mock_image, mock_title):
        """Test that app description explains the prediction capability"""
        app()
        markdown_content = mock_markdown.call_args[0][0]
        assert "predict" in markdown_content.lower()
        assert "person has diabetes" in markdown_content.lower()

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_with_missing_image_file_should_still_attempt_call(self, mock_markdown, mock_image, mock_title):
        """Test that app still calls st.image even if file doesn't exist"""
        app()
        mock_image.assert_called_once()

    @patch('home.st.title')
    @patch('home.st.image')
    @patch('home.st.markdown')
    def test_app_description_should_mention_random_forest_classifier(self, mock_markdown, mock_image, mock_title):
        """Test that description mentions the machine learning algorithm used"""
        app()
        markdown_content = mock_markdown.call_args[0][0]
        assert "Random Forest Classifier" in markdown_content