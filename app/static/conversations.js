/**
 * Gestion des conversations - Module réutilisable
 */

class ConversationsManager {
    constructor(projectId = null) {
        this.projectId = projectId;
        this.currentConversation = null;
        this.conversations = [];
        // Déterminer le nom du gestionnaire pour les callbacks
        this.managerName = projectId
            ? "projectConversationsManager"
            : "conversationsManager";
    }

    /**
     * Charger les conversations
     */
    async loadConversations() {
        const url = this.projectId
            ? `/api/conversations?project_id=${this.projectId}`
            : `/api/conversations`;

        const response = await apiCall(url);
        if (response && response.ok) {
            this.conversations = await response.json();
            this.renderConversationsList();
            return this.conversations;
        }
        return [];
    }

    /**
     * Créer une nouvelle conversation
     */
    async createConversation(title = "Nouvelle conversation") {
        const response = await apiCall("/api/conversations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                title,
                project_id: this.projectId,
            }),
        });

        if (response && response.ok) {
            const conversation = await response.json();
            this.conversations.unshift(conversation);
            this.selectConversation(conversation.id);
            this.renderConversationsList();
            return conversation;
        }
        return null;
    }

    /**
     * Sélectionner une conversation
     */
    async selectConversation(conversationId) {
        // Si on quitte une conversation qui a encore le titre par défaut, générer un titre automatiquement
        if (this.currentConversation && this.currentConversation !== conversationId) {
            const previousConversation = this.conversations.find(
                c => c.id === this.currentConversation
            );
            console.log("Conversation précédente:", previousConversation);
            // Générer un titre si c'est "Nouvelle conversation" ou un titre auto-généré (long ou se termine par "...")
            const shouldGenerateTitle = previousConversation && (
                previousConversation.title === "Nouvelle conversation" ||
                previousConversation.title === null ||
                previousConversation.title.length > 30 ||
                previousConversation.title.endsWith("...")
            );
            if (shouldGenerateTitle) {
                console.log("Génération du titre pour:", this.currentConversation);
                // Générer un titre automatiquement en arrière-plan (ne pas bloquer)
                this.generateTitleForConversation(this.currentConversation).catch(err => {
                    console.warn("Erreur lors de la génération du titre:", err);
                });
            }
        }

        this.currentConversation = conversationId;

        // Ouvrir le chatbot s'il est fermé
        this.openChatbotIfClosed();

        // Charger les messages de la conversation
        const response = await apiCall(
            `/api/conversations/${conversationId}/messages`
        );
        if (response && response.ok) {
            const messages = await response.json();
            this.renderMessages(messages);
            this.updateActiveConversation(conversationId);
            this.updateChatHistory(messages);
        }
    }

    /**
     * Générer automatiquement un titre pour une conversation
     */
    async generateTitleForConversation(conversationId) {
        console.log("Génération du titre pour la conversation:", conversationId);
        try {
            const response = await apiCall(
                `/api/conversations/${conversationId}/generate-title`,
                {
                    method: "POST",
                }
            );
            if (response && response.ok) {
                const updatedConversation = await response.json();
                console.log("Titre généré:", updatedConversation.title);
                // Mettre à jour la conversation dans la liste
                const index = this.conversations.findIndex(c => c.id === conversationId);
                if (index !== -1) {
                    this.conversations[index] = updatedConversation;
                    this.renderConversationsList();
                }
                return updatedConversation;
            } else {
                const errorText = await response.text();
                console.error("Erreur lors de la génération du titre:", response.status, errorText);
            }
        } catch (error) {
            console.error("Exception lors de la génération du titre:", error);
        }
        return null;
    }

    /**
     * Ouvrir le chatbot s'il est fermé
     */
    openChatbotIfClosed() {
        // Détecter si on est sur la page d'accueil ou la page projet
        const chatbotAreaHome = document.getElementById("chatbot-area");
        const chatbotAreaProject = document.getElementById(
            "project-chatbot-area"
        );
        const chatbotArea = chatbotAreaHome || chatbotAreaProject;

        if (!chatbotArea) return;

        // Si le chatbot est caché, l'ouvrir
        if (chatbotArea.classList.contains("hidden")) {
            // Appeler la fonction toggle appropriée
            if (chatbotAreaHome && typeof toggleChatbot === "function") {
                // Page d'accueil - vérifier l'état global
                if (typeof chatbotOpen !== "undefined" && !chatbotOpen) {
                    toggleChatbot();
                } else if (typeof chatbotOpen === "undefined") {
                    // Si la variable n'existe pas, forcer l'ouverture
                    toggleChatbot();
                }
            } else if (
                chatbotAreaProject &&
                typeof toggleProjectChatbot === "function"
            ) {
                // Page projet - vérifier l'état global
                if (
                    typeof projectChatbotOpen !== "undefined" &&
                    !projectChatbotOpen
                ) {
                    toggleProjectChatbot();
                } else if (typeof projectChatbotOpen === "undefined") {
                    // Si la variable n'existe pas, forcer l'ouverture
                    toggleProjectChatbot();
                }
            }
        }
    }

    /**
     * Mettre à jour l'historique de chat avec les messages chargés
     * IMPORTANT: Réinitialise complètement l'historique (ne pas ajouter à l'existant)
     */
    updateChatHistory(messages) {
        // Détecter si on est sur la page d'accueil ou la page projet
        // Réinitialiser complètement l'historique pour éviter le mélange des messages
        if (typeof chatHistory !== "undefined") {
            // Page d'accueil - réinitialiser complètement
            chatHistory.length = 0; // Vider d'abord
            chatHistory.push(
                ...messages.map((msg) => ({
                    role: msg.role,
                    content: msg.content,
                }))
            );
        } else if (typeof projectChatHistory !== "undefined") {
            // Page projet - réinitialiser complètement
            projectChatHistory.length = 0; // Vider d'abord
            projectChatHistory.push(
                ...messages.map((msg) => ({
                    role: msg.role,
                    content: msg.content,
                }))
            );
        }
    }

    /**
     * Supprimer une conversation
     */
    async deleteConversation(conversationId) {
        if (!confirm("Voulez-vous vraiment supprimer cette conversation ?")) {
            return;
        }

        const response = await apiCall(`/api/conversations/${conversationId}`, {
            method: "DELETE",
        });

        if (response && response.ok) {
            this.conversations = this.conversations.filter(
                (c) => c.id !== conversationId
            );

            if (this.currentConversation === conversationId) {
                this.currentConversation = null;
                this.clearMessages();
            }

            this.renderConversationsList();
        }
    }

    /**
     * Renommer une conversation
     */
    async renameConversation(conversationId, newTitle) {
        const response = await apiCall(`/api/conversations/${conversationId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: newTitle }),
        });

        if (response && response.ok) {
            const updated = await response.json();
            const index = this.conversations.findIndex(
                (c) => c.id === conversationId
            );
            if (index !== -1) {
                this.conversations[index] = updated;
                this.renderConversationsList();
            }
        }
    }

    /**
     * Rendre la liste des conversations
     */
    renderConversationsList() {
        // Chercher dans la sidebar de base.html (utilisée sur toutes les pages)
        const container = document.getElementById("conversations-list");
        if (!container) return;

        if (this.conversations.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 dark:text-gray-400 py-4 text-xs px-3">
                    <p>Aucune conversation</p>
                </div>
            `;
            return;
        }

        // Style adapté pour la sidebar de base.html (comme les projets)
        container.innerHTML = this.conversations
            .map(
                (conv) => `
            <div 
                class="group flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                    this.currentConversation === conv.id
                        ? "bg-blue-50 dark:bg-blue-900/20"
                        : "hover:bg-gray-100 dark:hover:bg-gray-700"
                }"
                onclick="${this.managerName}.selectConversation(${conv.id})"
            >
                <a href="#" class="flex-1 text-gray-700 dark:text-gray-300 cursor-pointer" onclick="event.preventDefault(); return false;">
                    <span class="block truncate text-sm">${this.escapeHtml(
                        conv.title
                    )}</span>
                    <span class="text-xs text-gray-500 dark:text-gray-400">${
                        conv.message_count || 0
                    } message${conv.message_count > 1 ? "s" : ""}</span>
                </a>
                <button
                    onclick="event.stopPropagation(); ${
                        this.managerName
                    }.showRenameDialog(${conv.id}, '${this.escapeHtml(
                    conv.title
                ).replace(/'/g, "\\'")}' )"
                    class="opacity-0 group-hover:opacity-100 text-blue-500 hover:text-blue-700 dark:hover:text-blue-400 p-1 transition-opacity"
                    title="Renommer"
                >
                    <i class="ti ti-pencil text-sm"></i>
                </button>
                <button
                    onclick="event.stopPropagation(); ${
                        this.managerName
                    }.deleteConversation(${conv.id})"
                    class="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-700 dark:hover:text-red-400 p-1 transition-opacity"
                    title="Supprimer"
                >
                    <i class="ti ti-trash text-sm"></i>
                </button>
            </div>
        `
            )
            .join("");
    }

    /**
     * Afficher le dialogue de renommage
     */
    showRenameDialog(conversationId, currentTitle) {
        const newTitle = prompt("Nouveau titre :", currentTitle);
        if (newTitle && newTitle.trim() && newTitle !== currentTitle) {
            this.renameConversation(conversationId, newTitle.trim());
        }
    }

    /**
     * Obtenir le conteneur de messages approprié (page d'accueil ou projet)
     */
    getChatMessagesContainer() {
        // Essayer d'abord le conteneur de la page projet
        let container = document.getElementById("project-chat-messages");
        if (!container) {
            // Sinon utiliser le conteneur de la page d'accueil
            container = document.getElementById("chat-messages");
        }
        return container;
    }

    /**
     * Rendre les messages dans le chat - EXACTEMENT comme dans les templates
     */
    renderMessages(messages) {
        const chatMessages = this.getChatMessagesContainer();
        if (!chatMessages) return;

        // TOUJOURS vider complètement le conteneur avant d'afficher les nouveaux messages
        // Cela évite le mélange des messages entre conversations
        chatMessages.innerHTML = "";

        if (messages.length === 0) {
            chatMessages.innerHTML = `
                <div class="text-center text-gray-500 dark:text-gray-400 py-8">
                    <i class="ti ti-message-circle text-3xl sm:text-4xl mb-2"></i>
                    <p class="text-sm sm:text-base">Commencez une conversation</p>
                </div>
            `;
            return;
        }

        // Créer les messages exactement comme dans les templates (addMessageToChat/addMessageToProjectChat)
        messages.forEach((msg) => {
            const messageDiv = document.createElement("div");
            messageDiv.className = `flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
            }`;

            // Pour les messages assistant, rendre le Markdown EXACTEMENT comme dans les templates
            let contentHtml;
            if (msg.role === "assistant") {
                const rawHtml = marked.parse(msg.content);
                const cleanHtml = DOMPurify.sanitize(rawHtml);
                // S'assurer que la classe markdown-content est bien appliquée pour les styles CSS
                contentHtml = `<div class="markdown-content">${cleanHtml}</div>`;
            } else {
                // Messages utilisateur en texte brut
                contentHtml = `<p class="whitespace-pre-wrap">${this.escapeHtml(
                    msg.content
                )}</p>`;
            }

            messageDiv.innerHTML = `
                <div class="max-w-2xl ${
                    msg.role === "user"
                        ? "bg-blue-600 dark:bg-blue-500 text-white"
                        : "bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 assistant-message"
                } rounded-lg px-4 py-2">
                    ${contentHtml}
                </div>
            `;
            chatMessages.appendChild(messageDiv);
        });

        // Scroller vers le bas après un court délai pour s'assurer que le DOM est mis à jour
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }

    /**
     * Effacer les messages
     */
    clearMessages() {
        const chatMessages = this.getChatMessagesContainer();
        if (chatMessages) {
            chatMessages.innerHTML = `
                <div class="text-center text-gray-500 dark:text-gray-400 py-8">
                    <i class="ti ti-message-circle text-3xl sm:text-4xl mb-2"></i>
                    <p class="text-sm sm:text-base">Commencez une conversation</p>
                </div>
            `;
        }
    }

    /**
     * Mettre à jour la conversation active visuellement
     */
    updateActiveConversation(conversationId) {
        document.querySelectorAll(".conversation-item").forEach((item) => {
            item.classList.remove(
                "bg-blue-50",
                "dark:bg-blue-900/20",
                "border-l-4",
                "border-blue-600"
            );
            item.classList.add("hover:bg-gray-50", "dark:hover:bg-gray-700/50");
        });

        const activeItem = document.querySelector(
            `.conversation-item[onclick*="${conversationId}"]`
        );
        if (activeItem) {
            activeItem.classList.remove(
                "hover:bg-gray-50",
                "dark:hover:bg-gray-700/50"
            );
            activeItem.classList.add(
                "bg-blue-50",
                "dark:bg-blue-900/20",
                "border-l-4",
                "border-blue-600"
            );
        }
    }

    /**
     * Échapper le HTML
     */
    escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }
}

// Exporter pour utilisation globale
window.ConversationsManager = ConversationsManager;
