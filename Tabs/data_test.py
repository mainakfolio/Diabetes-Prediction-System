import pytest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
from data import app


class TestDataApp:
    """Tests for the data module's app function"""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing"""
        return pd.DataFrame({
            'Age': [25, 30, 35, 40],
            'Glucose': [120, 130, 140, 150],
            'BloodPressure': [80, 85, 90, 95]
        })

    @pytest.fixture
    def mock_streamlit(self):
        """Mock streamlit module"""
        with patch('data.st') as mock_st:
            mock_st.title = Mock()
            mock_st.subheader = Mock()
            mock_st.expander = MagicMock()
            mock_st.checkbox = Mock(return_value=False)
            mock_st.columns = Mock(return_value=(Mock(), Mock(), Mock()))
            mock_st.selectbox = Mock()
            mock_st.dataframe = Mock()
            mock_st.markdown = Mock()
            yield mock_st

    def test_app_should_set_title_when_called(self, sample_dataframe, mock_streamlit):
        """Should set the page title to 'Data Info page'"""
        app(sample_dataframe)
        mock_streamlit.title.assert_called_once_with("Data Info page")

    def test_app_should_create_view_data_subheader(self, sample_dataframe, mock_streamlit):
        """Should create 'View Data' subheader"""
        app(sample_dataframe)
        calls = mock_streamlit.subheader.call_args_list
        assert any(call[0][0] == "View Data" for call in calls)

    def test_app_should_create_columns_description_subheader(self, sample_dataframe, mock_streamlit):
        """Should create 'Columns Description:' subheader"""
        app(sample_dataframe)
        calls = mock_streamlit.subheader.call_args_list
        assert any(call[0][0] == "Columns Description:" for call in calls)

    def test_app_should_display_dataframe_in_expander(self, sample_dataframe, mock_streamlit):
        """Should create expander with dataframe display"""
        mock_expander = MagicMock()
        mock_streamlit.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_streamlit.expander.return_value.__exit__ = Mock(return_value=False)
        
        app(sample_dataframe)
        
        mock_streamlit.expander.assert_called_once_with("View data")
        mock_expander.dataframe.assert_called_once_with(sample_dataframe)

    def test_app_should_show_summary_when_checkbox_selected(self, sample_dataframe, mock_streamlit):
        """Should display dataframe summary when 'View Summary' checkbox is checked"""
        mock_streamlit.checkbox.side_effect = [True, False, False, False]
        
        app(sample_dataframe)
        
        calls = mock_streamlit.dataframe.call_args_list
        summary_called = any(
            isinstance(call[0][0], pd.core.frame.DataFrame) and 
            call[0][0].equals(sample_dataframe.describe())
            for call in calls
        )
        assert summary_called

    def test_app_should_not_show_summary_when_checkbox_unchecked(self, sample_dataframe, mock_streamlit):
        """Should not display summary when 'View Summary' checkbox is unchecked"""
        mock_streamlit.checkbox.return_value = False
        
        app(sample_dataframe)
        
        calls = mock_streamlit.dataframe.call_args_list
        summary_called = any(
            isinstance(call[0][0], pd.core.frame.DataFrame) and 
            call[0][0].equals(sample_dataframe.describe())
            for call in calls
        )
        assert not summary_called

    def test_app_should_show_column_names_when_checkbox_selected(self, sample_dataframe, mock_streamlit):
        """Should display column names when 'Column Names' checkbox is checked"""
        mock_streamlit.checkbox.side_effect = [False, True, False, False]
        col1, col2, col3 = Mock(), Mock(), Mock()
        mock_streamlit.columns.return_value = (col1, col2, col3)
        
        app(sample_dataframe)
        
        calls = mock_streamlit.dataframe.call_args_list
        columns_called = any(
            hasattr(call[0][0], 'equals') and 
            call[0][0].equals(sample_dataframe.columns)
            for call in calls
        )
        assert columns_called

    def test_app_should_show_column_dtypes_when_checkbox_selected(self, sample_dataframe, mock_streamlit):
        """Should display column data types when 'Columns data types' checkbox is checked"""
        mock_streamlit.checkbox.side_effect = [False, False, True, False]
        col1, col2, col3 = Mock(), Mock(), Mock()
        mock_streamlit.columns.return_value = (col1, col2, col3)
        
        app(sample_dataframe)
        
        assert mock_streamlit.dataframe.called

    def test_app_should_show_column_data_when_checkbox_selected(self, sample_dataframe, mock_streamlit):
        """Should display specific column data when 'Columns Data' checkbox is checked"""
        mock_streamlit.checkbox.side_effect = [False, False, False, True]
        mock_streamlit.selectbox.return_value = 'Age'
        col1, col2, col3 = Mock(), Mock(), Mock()
        mock_streamlit.columns.return_value = (col1, col2, col3)
        
        app(sample_dataframe)
        
        assert mock_streamlit.selectbox.called

    def test_app_should_create_three_columns_layout(self, sample_dataframe, mock_streamlit):
        """Should create three columns for checkbox layout"""
        app(sample_dataframe)
        mock_streamlit.columns.assert_called_once_with(3)

    def test_app_should_display_kaggle_dataset_link(self, sample_dataframe, mock_streamlit):
        """Should display markdown with Kaggle dataset link"""
        app(sample_dataframe)
        
        markdown_calls = mock_streamlit.markdown.call_args_list
        assert len(markdown_calls) > 0
        markdown_content = markdown_calls[0][0][0]
        assert "kaggle.com" in markdown_content
        assert "pima-indians-diabetes-database" in markdown_content

    def test_app_should_use_unsafe_html_in_markdown(self, sample_dataframe, mock_streamlit):
        """Should allow unsafe HTML in markdown link"""
        app(sample_dataframe)
        
        markdown_calls = mock_stream