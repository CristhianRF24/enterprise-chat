import unittest
from unittest.mock import MagicMock
from sqlalchemy.orm import Session
from app.crud.file_crud import create_file, file_exists
from app.models.file import File

class TestFileFunctions(unittest.TestCase):
    
    # This method runs before each test
    def setUp(self):
        self.mock_db = MagicMock(spec=Session)

    # Test case to create a file in the database
    def test_create_file(self):
        filename = "test_file.txt"
        filetype = "text/plain"
        filepath = "/path/to/test_file.txt"
        
        result = create_file(self.mock_db, filename, filetype, filepath)
        
        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_called_once()
        
        self.assertEqual(result.filename, filename)
        self.assertEqual(result.filetype, filetype)
        self.assertEqual(result.filepath, filepath)

    # Test case to check if a file exists in the database
    def test_file_exists(self):
        filename = "existing_file.txt"
        filepath = "/path/to/existing_file.txt"
        mock_file = File(filename=filename, filepath=filepath)
        
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_file
        
        result = file_exists(self.mock_db, filename, filepath)
        
        self.assertEqual(result, mock_file)

    # Test case to check that a non-existing file returns None
    def test_file_does_not_exist(self):
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = file_exists(self.mock_db, "non_existing_file.txt", "/path/to/non_existing_file.txt")
        
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
