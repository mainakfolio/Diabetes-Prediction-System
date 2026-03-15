import pytest
from unittest.mock import patch, MagicMock, mock_open
from about import app


class TestAboutApp:
    
    def test_app_should_call_balloons_when_invoked(self):
        """Test that app() calls st.balloons()"""
        with patch('about.st') as mock_st:
            app()
            mock_st.balloons.assert_called_once()
    
    def test_app_should_set_title_to_contact_us(self):
        """Test that app() sets the page title to 'Contact Us'"""
        with patch('about.st') as mock_st:
            app()
            mock_st.title.assert_called_once_with('Contact Us')
    
    def test_app_should_display_name_markdown(self):
        """Test that app() displays the name in markdown format"""
        with patch('about.st') as mock_st:
            app()
            calls = mock_st.markdown.call_args_list
            assert any('Mainak Chaudhuri' in str(call) for call in calls)
    
    def test_app_should_display_description_markdown(self):
        """Test that app() displays the description"""
        with patch('about.st') as mock_st:
            app()
            calls = mock_st.markdown.call_args_list
            assert any('Passionate Engineer' in str(call) for call in calls)
    
    def test_app_should_display_image(self):
        """Test that app() displays an image"""
        with patch('about.st') as mock_st:
            with patch('about.Image'):
                app()
                mock_st.image.assert_called_once_with('./images/icon.jpg')
    
    def test_app_should_display_linkedin_link(self):
        """Test that app() displays LinkedIn link"""
        with patch('about.st') as mock_st:
            app()
            calls = mock_st.markdown.call_args_list
            assert any('linkedin.com' in str(call) for call in calls)
    
    def test_app_should_display_github_link(self):
        """Test that app() displays GitHub link"""
        with patch('about.st') as mock_st:
            app()
            calls = mock_st.markdown.call_args_list
            assert any('github.com' in str(call) for call in calls)
    
    def test_app_should_handle_multiple_markdown_calls(self):
        """Test that app() makes multiple markdown calls in sequence"""
        with patch('about.st') as mock_st:
            app()
            assert mock_st.markdown.call_count >= 4
    
    def test_app_should_call_all_streamlit_functions_once(self):
        """Test that app() executes without errors and calls expected functions"""
        with patch('about.st') as mock_st:
            with patch('about.Image'):
                app()
                assert mock_st.balloons.called
                assert mock_st.title.called
                assert mock_st.markdown.called
                assert mock_st.image.called
    
    def test_app_with_missing_image_file_should_still_call_image(self):
        """Test that app() still calls st.image even if file doesn't exist"""
        with patch('about.st') as mock_st:
            with patch('about.Image', side_effect=FileNotFoundError):
                app()
                mock_st.image.assert_called_once_with('./images/icon.jpg')
    
    def test_app_balloons_called_before_title(self):
        """Test that balloons is called before title"""
        with patch('about.st') as mock_st:
            app()
            call_order = [call[0] for call in mock_st.method_calls]
            balloons_index = next(i for i, call in enumerate(call_order) if call == 'balloons')
            title_index = next(i for i, call in enumerate(call_order) if call == 'title')
            assert balloons_index < title_index
    
    def test_app_markdown_contains_correct_linkedin_url(self):
        """Test that the LinkedIn markdown contains the correct URL"""
        with patch('about.st') as mock_st:
            app()
            linkedin_call = [call for call in mock_st.markdown.call_args_list 
                           if 'linkedin' in str(call).lower()][0]
            assert 'linkedin.com/in/mainak-chaudhuri-127898176' in str(linkedin_call)
    
    def test_app_markdown_contains_correct_github_url(self):
        """Test that the GitHub markdown contains the correct URL"""
        with patch('about.st') as mock_st:
            app()
            github_call = [call for call in mock_st.markdown.call_args_list 
                          if 'github' in str(call).lower()][0]
            assert 'github.com/MainakRepositor' in str(github_call)
    
    def test_app_executes_without_exceptions(self):
        """Test that app() runs without raising any exceptions"""
        with patch('about.st') as mock_st:
            with patch('about.Image'):
                try:
                    app()
                except Exception as e:
                    pytest.fail(f"app() raised {type(e).__name__} unexpectedly: {e}")
    
    def test_app_image_path_is_correct(self):
        """Test that app() uses the correct image path"""
        with patch('about.st') as mock_st:
            app()
            mock_st.image.assert_called_with('./images/icon.jpg')