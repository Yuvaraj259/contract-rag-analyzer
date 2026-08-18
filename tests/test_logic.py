import unittest
from src.query_parser import parse_query, detect_operator, extract_requested_skills
from src.rag_service import is_candidate_match, calculate_evidence_strength

class TestLogic(unittest.TestCase):
    def test_query_parser_operator(self):
        self.assertEqual(detect_operator("Which candidates have worked with React or Angular?"), "OR")
        self.assertEqual(detect_operator("Find candidates who know React and Angular"), "AND")
        self.assertEqual(detect_operator("Who knows React?"), "OR")

    def test_query_parser_skills(self):
        skills = extract_requested_skills("Which candidates have worked with React or Angular?")
        self.assertIn("React", skills)
        self.assertIn("Angular", skills)

    def test_query_parser_ranking(self):
        self.assertTrue(parse_query("Who is the strongest React candidate?")["ranking_requested"])
        self.assertFalse(parse_query("Which candidates have worked with React?")["ranking_requested"])

    def test_evidence_strength(self):
        self.assertEqual(calculate_evidence_strength("professional_experience", 2), "strong")
        self.assertEqual(calculate_evidence_strength("internship", 1), "moderate")
        self.assertEqual(calculate_evidence_strength("project", 1), "limited")
        self.assertEqual(calculate_evidence_strength("skills_section_only", 1), "explicitly_mentioned")

    def test_is_candidate_match_or(self):
        skill_results = {
            "React": {"status": "CONFIRMED"},
            "Angular": {"status": "NOT_CONFIRMED"}
        }
        self.assertTrue(is_candidate_match(skill_results, "OR"))

        skill_results_only_angular = {
            "React": {"status": "NOT_CONFIRMED"},
            "Angular": {"status": "CONFIRMED"}
        }
        self.assertTrue(is_candidate_match(skill_results_only_angular, "OR"))
        
        skill_results_neither = {
            "React": {"status": "NOT_CONFIRMED"},
            "Angular": {"status": "NOT_CONFIRMED"}
        }
        self.assertFalse(is_candidate_match(skill_results_neither, "OR"))

    def test_is_candidate_match_and(self):
        skill_results = {
            "React": {"status": "CONFIRMED"},
            "Angular": {"status": "CONFIRMED"}
        }
        self.assertTrue(is_candidate_match(skill_results, "AND"))
        
        skill_results_missing_one = {
            "React": {"status": "CONFIRMED"},
            "Angular": {"status": "NOT_CONFIRMED"}
        }
        self.assertFalse(is_candidate_match(skill_results_missing_one, "AND"))

if __name__ == "__main__":
    unittest.main()
