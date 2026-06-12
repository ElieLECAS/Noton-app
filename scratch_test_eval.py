import asyncio
import os
import sys

# Ajouter le chemin de l'application
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from sqlmodel import Session
from app.database import engine
from app.services.retriever_evaluator import generate_rag_response, run_llm_judge
from app.config import settings

async def test_judge():
    print("=== TEST LLM JUDGE ===")
    question = "À quelle fréquence doit-on planifier la maintenance ?"
    generated = "La maintenance de sécurité pour les écoles et hôtels doit être effectuée au moins une fois par an."
    expected = "La maintenance de sécurité doit être effectuée au moins une fois par an pour les établissements scolaires ou hôteliers."
    
    print(f"API Key présente: {bool(settings.MISTRAL_API_KEY)}")
    print(f"Provider LLM: {settings.LLM_PROVIDER}")
    print(f"Modèle Fast: {settings.MODEL_FAST}")
    
    try:
        res = await run_llm_judge(
            question=question,
            generated_response=generated,
            expected_response=expected,
            judge_model="mistral-small-latest"
        )
        print("Résultat du juge :")
        print(res)
    except Exception as e:
        print(f"Erreur lors du test du juge : {e}")

async def test_generation():
    print("\n=== TEST GENERATION RAG ===")
    # Récupérer un espace existant pour tester s'il y a des passages
    from app.models.space import Space
    from app.services.space_search_service import search_relevant_passages
    
    with Session(engine) as session:
        spaces = session.exec(Space).all()
        if not spaces:
            print("Aucun espace trouvé dans la base de données. Impossible de tester la génération.")
            return
        
        space = spaces[0]
        print(f"Espace de test : {space.name} (ID: {space.id})")
        
        # Faire une recherche bidon
        question = "tension de réglage"
        search_res = await search_relevant_passages(
            session=session,
            space_id=space.id,
            query_text=question,
            user_id=1,
            k=3,
            document_filter="all"
        )
        passages = search_res.get("passages", [])
        print(f"Passages trouvés : {len(passages)}")
        
        # Tester la génération
        response = await generate_rag_response(
            session=session,
            space_id=space.id,
            user_id=1,
            question=question,
            passages=passages
        )
        print("Réponse générée :")
        print(response)

async def main():
    await test_judge()
    await test_generation()

if __name__ == "__main__":
    asyncio.run(main())
