#pragma once
#include <string>
#include <memory>
#include <cmath>
#include <queue>
#include <openssl/evp.h>
#include <fstream>
#include <stdexcept>

struct MerkleNode
{
	std::unique_ptr<MerkleNode> left;
	std::unique_ptr<MerkleNode> right;
    std::array<uint8_t, 32> hash;
    bool is_set = false;
};

using HashBuffer = std::array<uint8_t, 32>;

//Computes a hash based on a given string
void computeHash(const std::string& data, MerkleNode* node)
{
    static thread_local EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    static const EVP_MD* md = EVP_get_digestbyname("sha256");

    unsigned int len = 0;
    EVP_DigestInit_ex(ctx, md, nullptr);
    EVP_DigestUpdate(ctx, data.data(), data.size());
    EVP_DigestFinal_ex(ctx, node->hash.data(), &len);

    node->is_set = true;
}

//Computes hash for internal nodes
void computeInternalHash(MerkleNode* parent, MerkleNode* left, MerkleNode* right) {
    static thread_local EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    static const EVP_MD* md = EVP_get_digestbyname("sha256");

    unsigned int len = 0;
    EVP_DigestInit_ex(ctx, md, nullptr);
    EVP_DigestUpdate(ctx, left->hash.data(), 32);
    EVP_DigestUpdate(ctx, right->hash.data(), 32);
    EVP_DigestFinal_ex(ctx, parent->hash.data(), &len);

    parent->is_set = true;
}

//Construct an empty tree with the necessary amount of nodes for hash tree
//Then populates leaf nodes with hashed data
void makeEmptyTree(int depth, int cur, MerkleNode* node, const std::vector<std::string>& data, size_t& dataIdx)
{
	if (cur < depth)
	{
        node->left = std::make_unique<MerkleNode>();
        node->right = std::make_unique<MerkleNode>();
        makeEmptyTree(depth, cur + 1, node->left.get(), data, dataIdx);
        makeEmptyTree(depth, cur + 1, node->right.get(), data, dataIdx);
	}
    if (cur == depth && dataIdx < data.size())
    {
        computeHash(data[dataIdx], node);
        dataIdx++;
    }
}

//Recursively computes hashes for internal nodes inside a Merkle-tree
//Requires that the overall tree structure is instantiated
//Leaf nodes of Merkle-tree should contain a hash otherwise root hash will be empty
void hashData(MerkleNode* node)
{
    if (node->left != nullptr)
    {
        hashData(node->left.get());
    }
    if (node->right != nullptr)
    {
        hashData(node->right.get());
    }

    if (node->left == nullptr && node->right == nullptr)
    {
        return;
    }
    else
    {
        static thread_local EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        static const EVP_MD* md = EVP_get_digestbyname("sha256");

        EVP_DigestInit_ex(ctx, md, nullptr);

        EVP_DigestUpdate(ctx, node->left->hash.data(), node->left->hash.size());

        EVP_DigestUpdate(ctx, node->right->hash.data(), node->right->hash.size());

        unsigned int len = 0;
        EVP_DigestFinal_ex(ctx, node->hash.data(), &len);

        node->is_set = true;
    }
}