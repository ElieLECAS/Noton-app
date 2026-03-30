"""Bibliothèque : upload, dossiers, documents, espaces (mocks traitement)."""
from __future__ import annotations

from unittest import mock


def test_upload_is_paid_true_persisted(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_paid.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("paid.txt", b"x", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "true"},
        )
    assert r.status_code == 201
    doc_id = r.json()[0]["id"]
    gr = client.get(f"/api/library/documents/{doc_id}", headers=responsable_headers)
    assert gr.status_code == 200
    assert gr.json()["is_paid"] is True


def test_upload_is_paid_false(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_free.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("free.txt", b"y", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    assert r.json()[0]["is_paid"] is False


def test_upload_invalid_space_ids_json(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/x.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("a.txt", b"z", "text/plain"))],
            data={"space_ids": "not-json", "is_paid": "false"},
        )
    assert r.status_code == 400


def test_upload_calls_process_once_per_file(client, responsable_headers):
    calls: list[tuple[int, str]] = []

    def _track(doc_id: int, path: str) -> None:
        calls.append((doc_id, path))

    with (
        mock.patch(
            "app.routers.library.process_document_async",
            side_effect=_track,
        ),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_multi.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[
                ("files", ("one.txt", b"1", "text/plain")),
                ("files", ("two.txt", b"2", "text/plain")),
            ],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    assert len(r.json()) == 2
    assert len(calls) == 2
    assert calls[0][0] != calls[1][0]


def test_folder_crud_and_move(client, responsable_headers):
    r = client.post(
        "/api/library/folders",
        headers=responsable_headers,
        json={"name": "Parent pytest", "parent_folder_id": None},
    )
    assert r.status_code == 201
    parent_id = r.json()["id"]

    r = client.post(
        "/api/library/folders",
        headers=responsable_headers,
        json={"name": "Enfant pytest", "parent_folder_id": parent_id},
    )
    assert r.status_code == 201
    child_id = r.json()["id"]

    r = client.put(
        f"/api/library/folders/{child_id}",
        headers=responsable_headers,
        json={"name": "Enfant renommé"},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "Enfant renommé"

    r = client.post(
        f"/api/library/folders/{child_id}/move",
        headers=responsable_headers,
    )
    assert r.status_code == 200
    assert r.json()["parent_folder_id"] is None

    r = client.delete(f"/api/library/folders/{child_id}", headers=responsable_headers)
    assert r.status_code == 204

    r = client.delete(f"/api/library/folders/{parent_id}", headers=responsable_headers)
    assert r.status_code == 204


def test_document_move_to_folder(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/move_me.pdf",
        ),
    ):
        up = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("moveme.txt", b"doc", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert up.status_code == 201
    doc_id = up.json()[0]["id"]

    fr = client.post(
        "/api/library/folders",
        headers=responsable_headers,
        json={"name": "Dossier cible", "parent_folder_id": None},
    )
    folder_id = fr.json()["id"]

    r = client.post(
        f"/api/library/documents/{doc_id}/move",
        headers=responsable_headers,
        params={"new_folder_id": folder_id},
    )
    assert r.status_code == 200
    assert r.json()["folder_id"] == folder_id

    client.delete(f"/api/library/documents/{doc_id}", headers=responsable_headers)
    client.delete(f"/api/library/folders/{folder_id}", headers=responsable_headers)


@mock.patch("app.routers.library.process_document_async")
@mock.patch(
    "app.routers.library.save_uploaded_file",
    return_value="media/documents/spaces_doc.pdf",
)
def test_document_add_and_remove_space(mock_save, mock_proc, client, responsable_headers):
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace doc link"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]

    up = client.post(
        "/api/library/upload",
        headers=responsable_headers,
        files=[("files", ("linked.txt", b"t", "text/plain"))],
        data={"space_ids": "[]", "is_paid": "false"},
    )
    assert up.status_code == 201
    doc_id = up.json()[0]["id"]

    r = client.post(
        f"/api/library/documents/{doc_id}/spaces",
        headers=responsable_headers,
        json={"add_space_ids": [space_id], "remove_space_ids": []},
    )
    assert r.status_code == 200

    lst = client.get(f"/api/library/documents/{doc_id}/spaces", headers=responsable_headers)
    assert lst.status_code == 200
    ids = {s["id"] for s in lst.json()}
    assert space_id in ids

    r = client.post(
        f"/api/library/documents/{doc_id}/spaces",
        headers=responsable_headers,
        json={"add_space_ids": [], "remove_space_ids": [space_id]},
    )
    assert r.status_code == 200

    lst2 = client.get(f"/api/library/documents/{doc_id}/spaces", headers=responsable_headers)
    assert space_id not in {s["id"] for s in lst2.json()}

    client.delete(f"/api/library/documents/{doc_id}", headers=responsable_headers)
    client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)
