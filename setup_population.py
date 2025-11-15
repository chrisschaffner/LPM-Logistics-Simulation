{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44924916",
   "metadata": {},
   "outputs": [
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mDer Kernel ist beim Ausführen von Code in der aktuellen Zelle oder einer vorherigen Zelle abgestürzt. \n",
      "\u001b[1;31mBitte überprüfen Sie den Code in der/den Zelle(n), um eine mögliche Fehlerursache zu identifizieren. \n",
      "\u001b[1;31mKlicken Sie <a href='https://aka.ms/vscodeJupyterKernelCrash'>hier</a>, um weitere Informationen zu erhalten. \n",
      "\u001b[1;31mWeitere Informationen finden Sie unter Jupyter <a href='command:jupyter.viewOutput'>Protokoll</a>."
     ]
    }
   ],
   "source": [
    "import os\n",
    "import pickle\n",
    "import torch\n",
    "\n",
    "# Number of agents\n",
    "num_agents = 10000\n",
    "\n",
    "# Folder to save your population\n",
    "save_dir = \"populations/population_10000\"\n",
    "os.makedirs(save_dir, exist_ok=True)\n",
    "\n",
    "# --- 1️⃣ Create current_edge property ---\n",
    "# Example: 3 values per agent (like a 3D position)\n",
    "current_edge = torch.zeros(num_agents, 3)  # shape: (10000, 3)\n",
    "\n",
    "with open(os.path.join(save_dir, \"current_edge.pickle\"), \"wb\") as f:\n",
    "    pickle.dump(current_edge, f)\n",
    "\n",
    "# --- 2️⃣ Create edge_progress property ---\n",
    "# Example: 1 value per agent\n",
    "edge_progress = torch.zeros(num_agents, 1)  # shape: (10000, 1)\n",
    "\n",
    "with open(os.path.join(save_dir, \"edge_progress.pickle\"), \"wb\") as f:\n",
    "    pickle.dump(edge_progress, f)\n",
    "\n",
    "print(f\"Population folder created at '{save_dir}' with 10000 agents.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "IDP",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
